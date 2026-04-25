"""POST /api/runs/new — trigger a pipeline run, stream events via /api/stream."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from auth import CurrentUser
from sim_config import settings
from events import publish, set_run_user
from metrics.cost import CostTracker, set_tracker
from models import preflight_db
from services.pipeline import persist_run, run_full_pipeline

router = APIRouter(tags=["control"])
logger = logging.getLogger(__name__)


_DEFAULT_BRIEF = """ProMeetings is a mobile app for structured, in-person, 1-on-1 professional
meetings — targeted at remote workers, conference attendees, business travellers,
and solo founders. Users publish a 2h-7d availability window, get matched with
another professional within 10km, and confirm the meetup via GPS check-in at a
café / restaurant / bar. The app is explicitly NOT a dating app.

Pricing: freemium. Personal tier free, limited to 3 scheduled + 1 instant meetups.
Professional tier 19€/month, 3 months free trial, unlimited meetups + pro profile
+ conference mode (temporary in-conference visibility)."""


class StartRunRequest(BaseModel):
    brief: str = Field(default=_DEFAULT_BRIEF)
    panel_size: int = Field(default=10, ge=3, le=50)
    rounds: int = Field(default=3, ge=1, le=3)


class StartRunResponse(BaseModel):
    status: str
    run_id: str


# In-flight tracking for the dev-local / no-DB path. Holds the set of
# user_ids whose pipeline is currently running. Guarded by `_local_lock`.
# When DATABASE_URL is set, the runs table is the source of truth instead
# (status='running' rows == in-flight pipelines) and this set is unused.
_local_inflight: set[str] = set()
_local_lock = asyncio.Lock()

# Idempotency cache: (user_id, key) → (run_id, expiry_ts). Network-flaky
# clients can safely retry POST /runs/new with the same Idempotency-Key
# header — the same key returns the original run_id instead of tripping
# the per-user 409. TTL kept short (5 min) so the cache stays small and
# stale keys can't shadow a legitimate new run with the same UUID.
_IDEMPOTENCY_TTL_S = 300.0
_idempotency_cache: dict[tuple[str, str], tuple[str, float]] = {}


def _idempotency_lookup(user_id: str, key: str | None) -> str | None:
    """Return the cached run_id for (user, key) if still valid, else None.
    Also opportunistically GC expired entries while we're walking the dict."""
    if not key:
        return None
    now = time.time()
    expired = [k for k, (_, exp) in _idempotency_cache.items() if exp <= now]
    for k in expired:
        _idempotency_cache.pop(k, None)
    hit = _idempotency_cache.get((user_id, key))
    return hit[0] if hit else None


def _idempotency_remember(user_id: str, key: str | None, run_id: str) -> None:
    if not key:
        return
    _idempotency_cache[(user_id, key)] = (run_id, time.time() + _IDEMPOTENCY_TTL_S)


def _do_run(
    brief: str, panel_size: int, rounds: int, run_id: str, user_id: str,
) -> None:
    """Sync worker — executed on a thread via asyncio.to_thread."""
    tracker = CostTracker(budget_usd=settings().budget_usd)
    set_tracker(tracker)
    try:
        result = run_full_pipeline(
            brief,
            panel_size=panel_size,
            rounds=rounds,
            run_id=run_id,
            user_id=user_id,
        )
    except Exception as e:
        logger.exception("pipeline failed")
        # Stamp the DB row terminal so the frontend sidebar shows an
        # 'error' chip instead of a forever-spinning 'running' card.
        preflight_db.update_run_terminal(
            run_id=run_id, status="error", error_message=str(e),
        )
        publish("run.error", {"run_id": run_id, "error": str(e)})
        return

    cost_summary = tracker.summary()
    try:
        paths = persist_run(result, cost_summary)
        publish(
            "run.persisted",
            {"run_id": run_id, "paths": {k: str(v) for k, v in paths.items()}},
        )
    except Exception as e:
        logger.exception("persist failed")
        preflight_db.update_run_terminal(
            run_id=run_id, status="error", error_message=f"persist: {e}",
        )
        publish("run.error", {"run_id": run_id, "error": f"persist: {e}"})


# Heartbeat cadence — long enough not to flood the SSE stream, short
# enough that a 30s+ silent stretch of OASIS simulation still ticks the
# UI's "running for X minutes" indicator. Tunable; not env-configurable
# yet because nobody's needed to.
_HEARTBEAT_INTERVAL_S = 10.0


async def _heartbeat_loop(run_id: str, user_id: str, started: float) -> None:
    """Publish run.heartbeat every N seconds with the elapsed time.

    The pipeline itself emits granular phase events, but the simulation
    phase can spend 30s+ between publishes — the UI looks frozen. This
    loop keeps a steady pulse on the bus so the live timeline shows
    activity even during silent stretches.
    """
    set_run_user(user_id)
    try:
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
            publish(
                "run.heartbeat",
                {
                    "run_id": run_id,
                    "elapsed_s": round(time.time() - started, 1),
                },
            )
    except asyncio.CancelledError:
        # Normal shutdown path when the worker thread finishes.
        raise


async def _run_and_release(
    brief: str, panel_size: int, rounds: int, run_id: str, user_id: str,
) -> None:
    """Run the pipeline on a worker thread and release the per-user
    in-memory slot when finished. The DB-side 'lock' (status='running')
    is released by `_do_run` itself when it stamps the terminal status.
    """
    hb_task = asyncio.create_task(
        _heartbeat_loop(run_id, user_id, time.time()),
        name=f"heartbeat-{run_id}",
    )
    try:
        await asyncio.to_thread(_do_run, brief, panel_size, rounds, run_id, user_id)
    finally:
        hb_task.cancel()
        # Awaiting the cancelled task keeps tracebacks out of the logs
        # when Python's task-destruction warnings would otherwise fire.
        try:
            await hb_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        async with _local_lock:
            _local_inflight.discard(user_id)


@router.post("/runs/new", response_model=StartRunResponse)
async def start_run(
    req: StartRunRequest,
    user: CurrentUser,
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key", max_length=128),
    ] = None,
) -> StartRunResponse:
    # Idempotency: a client retrying after a network blip should land
    # back on the same run_id, not trip the per-user 409 below. Looked
    # up *before* the concurrency check so a retry never has to claim a
    # slot it already owns.
    cached_run_id = _idempotency_lookup(user, idempotency_key)
    if cached_run_id is not None:
        return StartRunResponse(status="started", run_id=cached_run_id)

    # Per-user concurrency. Two layers:
    #   - in-memory set under _local_lock — race-proof inside a single
    #     process and works in dev-local mode (no DATABASE_URL).
    #   - has_running_run_for_user — checked while holding the local lock
    #     so the DB row is the cross-process truth (one replica racing
    #     another still narrows to a tiny window before insert_run lands).
    # _do_run releases the DB row by stamping status terminal; the
    # asyncio finally hook releases the local set entry.
    run_id = str(uuid4())
    async with _local_lock:
        if user in _local_inflight:
            raise HTTPException(
                status_code=409,
                detail="You already have a run in progress — wait for it to finish.",
            )
        if preflight_db.has_running_run_for_user(user) is True:
            raise HTTPException(
                status_code=409,
                detail="You already have a run in progress — wait for it to finish.",
            )
        _local_inflight.add(user)
        # Insert the runs row up-front (still under the lock) so the next
        # request from this user trips has_running_run_for_user. If the
        # DB is unavailable this no-ops and _local_inflight alone carries
        # the concurrency guarantee.
        preflight_db.insert_run(
            run_id=run_id,
            auth_uid=user,
            brief=req.brief,
            panel_size=req.panel_size,
            rounds=req.rounds,
            settings={"simulation_seed": 42},
        )

    asyncio.create_task(
        _run_and_release(req.brief, req.panel_size, req.rounds, run_id, user)
    )
    _idempotency_remember(user, idempotency_key, run_id)
    return StartRunResponse(status="started", run_id=run_id)


class CancelRunResponse(BaseModel):
    status: str
    run_id: str


@router.post("/runs/{run_id}/cancel", response_model=CancelRunResponse)
async def cancel_run(run_id: str, user: CurrentUser) -> CancelRunResponse:
    """Soft-cancel an in-flight run.

    Marks the DB row as error('cancelled by user') and drops the in-memory
    inflight slot so the user can start a new run immediately. The worker
    thread itself keeps going (no inter-thread cancellation primitives
    wired up) but its eventual `update_run_terminal('done')` is gated on
    status='running' and so can't overwrite the cancellation.
    """
    cancelled = preflight_db.cancel_run(run_id=run_id, auth_uid=user)
    if cancelled is None:
        # DB unavailable — fall through and rely on the local set so
        # at least dev-local UX works.
        async with _local_lock:
            was_inflight = user in _local_inflight
            _local_inflight.discard(user)
        if not was_inflight:
            raise HTTPException(
                status_code=404,
                detail="No run in progress to cancel.",
            )
        set_run_user(user)
        publish("run.cancelled", {"run_id": run_id})
        return CancelRunResponse(status="cancelled", run_id=run_id)

    if not cancelled:
        # The DB knows about the run but it's already terminal (or the
        # caller doesn't own it).
        raise HTTPException(
            status_code=409,
            detail="Run is not in progress (already finished, errored, or not yours).",
        )

    async with _local_lock:
        _local_inflight.discard(user)
    set_run_user(user)
    publish("run.cancelled", {"run_id": run_id})
    return CancelRunResponse(status="cancelled", run_id=run_id)

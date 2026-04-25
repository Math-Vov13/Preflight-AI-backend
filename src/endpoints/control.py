"""POST /api/runs/new — trigger a pipeline run, stream events via /api/stream."""
from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from auth import CurrentUser
from sim_config import settings
from events import publish
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


async def _run_and_release(
    brief: str, panel_size: int, rounds: int, run_id: str, user_id: str,
) -> None:
    """Run the pipeline on a worker thread and release the per-user
    in-memory slot when finished. The DB-side 'lock' (status='running')
    is released by `_do_run` itself when it stamps the terminal status.
    """
    try:
        await asyncio.to_thread(_do_run, brief, panel_size, rounds, run_id, user_id)
    finally:
        async with _local_lock:
            _local_inflight.discard(user_id)


@router.post("/runs/new", response_model=StartRunResponse)
async def start_run(req: StartRunRequest, user: CurrentUser) -> StartRunResponse:
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
    return StartRunResponse(status="started", run_id=run_id)

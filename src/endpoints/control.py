"""POST /api/runs/new — trigger a pipeline run, stream events via /api/stream."""
from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from auth import CurrentUser
from sim_config import settings
from events import publish
from metrics.cost import CostTracker, set_tracker
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


_running: bool = False
_lock = asyncio.Lock()


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
        publish("run.error", {"run_id": run_id, "error": f"persist: {e}"})


async def _run_and_release(
    brief: str, panel_size: int, rounds: int, run_id: str, user_id: str,
) -> None:
    global _running
    try:
        await asyncio.to_thread(_do_run, brief, panel_size, rounds, run_id, user_id)
    finally:
        async with _lock:
            _running = False


@router.post("/runs/new", response_model=StartRunResponse)
async def start_run(req: StartRunRequest, user: CurrentUser) -> StartRunResponse:
    # Global lock — only one run can be in flight at a time across all
    # users. Acceptable at hackathon scale; for a real multi-tenant
    # deployment this would become per-user (post-hack TODO in
    # docs/roadmap-post-hack.md, item D).
    global _running
    async with _lock:
        if _running:
            raise HTTPException(
                status_code=409,
                detail="A run is already in progress — wait for it to finish.",
            )
        _running = True
    run_id = time.strftime("%Y%m%d_%H%M%S")
    asyncio.create_task(
        _run_and_release(req.brief, req.panel_size, req.rounds, run_id, user)
    )
    return StartRunResponse(status="started", run_id=run_id)

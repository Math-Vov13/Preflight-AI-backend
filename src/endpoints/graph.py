"""Graph search endpoint — exposes the Zep knowledge graph to the frontend.

This is the read-side of `services/zep_memory.py`. Ingestion happens
automatically at the end of every run; this endpoint lets the UI and the
chat-context builder query facts (edges) or entities (nodes) across *all*
runs.

Returns 503 when ZEP_API_KEY is unset — the product remains usable without
it, but cross-run memory queries degrade gracefully.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query

from auth import CurrentUser
from services.zep_memory import get_memory, is_enabled

router = APIRouter(tags=["graph"])
logger = logging.getLogger(__name__)


@router.get("/graph/search")
def graph_search(
    user: CurrentUser,
    q: str = Query(..., min_length=2, max_length=300, description="natural-language query"),
    scope: Literal["edges", "nodes"] = Query("edges"),
    limit: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    if not is_enabled():
        raise HTTPException(
            status_code=503,
            detail="Zep memory is disabled — set ZEP_API_KEY to enable graph search.",
        )

    memory = get_memory()
    assert memory is not None  # is_enabled guarantees this

    try:
        results = memory.search(q, user_id=user, scope=scope, limit=limit)
    except Exception as e:  # noqa: BLE001
        logger.exception("graph search failed for q=%r", q)
        raise HTTPException(status_code=502, detail=f"Zep search failed: {e}") from e

    return {"scope": scope, "query": q, "count": len(results), "results": results}


@router.get("/graph/status")
def graph_status() -> dict[str, Any]:
    """Lightweight enablement probe — lets the UI decide whether to hint about
    graph features without spending an API call."""
    return {"enabled": is_enabled()}

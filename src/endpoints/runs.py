"""Run listing and fetch endpoints, scoped to the authenticated user.

DB-first (BE-PR1): when DATABASE_URL is set and the caller's auth_uid is
a real Supabase UUID we read from the `runs` + `run_artifacts` tables.
On any miss (DB unavailable, dev-local user, run not in DB yet) we fall
back to the legacy on-disk JSON layout under data/runs/{user_id}/. Each
mode produces the *same* response shape so the frontend doesn't have to
care.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from auth import CurrentUser
from models import preflight_db
from paths import user_runs_dir

router = APIRouter(tags=["runs"])

# File naming convention inherited from ai-agent-demo; may rename to "run_" later.
_FILE_PREFIX = "pre_demo_"


def _run_id(p: Path) -> str:
    stem = p.stem
    return stem[len(_FILE_PREFIX):] if stem.startswith(_FILE_PREFIX) else stem


def _load_json(p: Path) -> dict[str, Any] | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _summary_from_db_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project a runs row into the listing shape the frontend expects."""
    return {
        "id": row["id"],
        "timestamp": row["id"],
        "brief_preview": (row.get("brief") or "")[:300],
        "panel_size": row.get("panel_size"),
        "rounds": row.get("rounds"),
        "total_latency_s": float(row["wall_s"]) if row.get("wall_s") is not None else None,
        "total_cost_usd": float(row["cost_usd"]) if row.get("cost_usd") is not None else None,
        "go_no_go_recommendation": row.get("verdict"),
    }


@router.get("/runs")
def list_runs(user: CurrentUser) -> list[dict[str, Any]]:
    # Try DB first. None means "DB unavailable for this user" — fall back
    # to file scan. Empty list means "DB available, no runs yet" — return
    # it as-is so we don't double-list anything that's also on disk.
    db_rows = preflight_db.list_runs_for_user(user)
    if db_rows is not None:
        return [_summary_from_db_row(r) for r in db_rows]

    # ── File-mode fallback ───────────────────────────────────────────
    runs_dir = user_runs_dir(user)
    if not runs_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for art_path in sorted(runs_dir.glob(f"{_FILE_PREFIX}*.json"), reverse=True):
        # Sidecar files share the prefix — only the *primary* artefact
        # (no underscore-suffix marker) is a run.
        if art_path.name.endswith(("_metrics.json", "_chat.json")):
            continue
        run_id = _run_id(art_path)
        metrics = _load_json(art_path.with_name(f"{art_path.stem}_metrics.json")) or {}
        run_block = metrics.get("run", {}) or {}
        cost_block = metrics.get("cost", {}) or {}
        out.append(
            {
                "id": run_id,
                "timestamp": run_id,
                **run_block,
                "total_cost_usd": cost_block.get("total_usd"),
                "calls": cost_block.get("calls"),
            }
        )
    return out


@router.get("/runs/{run_id}")
def get_run(run_id: str, user: CurrentUser) -> dict[str, Any]:
    db_run = preflight_db.get_run_with_artifacts(run_id=run_id, auth_uid=user)
    if db_run is not None:
        # Build the same {metrics, artefacts} shape the file path produces.
        artefacts_by_kind: dict[str, Any] = {
            kind: entry.get("payload") for kind, entry in db_run.get("artifacts", {}).items()
        }
        # The "panel" artefact stores {"personas": [...]} — frontend expects
        # the bare list under artefacts.panel. Unwrap defensively.
        panel = artefacts_by_kind.get("panel")
        if isinstance(panel, dict) and "personas" in panel:
            panel = panel["personas"]
        return {
            "id": run_id,
            "timestamp": run_id,
            "metrics": {
                "run": {
                    "run_id": run_id,
                    "brief_preview": (db_run.get("brief") or "")[:300],
                    "panel_size": db_run.get("panel_size"),
                    "rounds": db_run.get("rounds"),
                    "total_latency_s": (
                        float(db_run["wall_s"]) if db_run.get("wall_s") is not None else None
                    ),
                    "go_no_go_recommendation": db_run.get("verdict"),
                },
                "cost": {
                    "total_usd": (
                        float(db_run["cost_usd"]) if db_run.get("cost_usd") is not None else None
                    ),
                },
                "judge": artefacts_by_kind.get("judge_scores"),
            },
            "artefacts": {
                "run_id": run_id,
                "brief": db_run.get("brief"),
                "ontology": artefacts_by_kind.get("ontology"),
                "panel": panel,
                "thread": artefacts_by_kind.get("thread"),
                "validation_report": artefacts_by_kind.get("validation_report"),
            },
        }

    # ── File-mode fallback ───────────────────────────────────────────
    art_path = user_runs_dir(user) / f"{_FILE_PREFIX}{run_id}.json"
    if not art_path.exists():
        # 404 — and we explicitly do NOT fall back to other users' dirs even
        # when the file exists elsewhere. Shielding that is the whole point
        # of the per-user namespace.
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    artefacts = _load_json(art_path) or {}
    metrics = _load_json(art_path.with_name(f"{art_path.stem}_metrics.json")) or {}
    return {
        "id": run_id,
        "timestamp": run_id,
        "metrics": metrics,
        "artefacts": artefacts,
    }

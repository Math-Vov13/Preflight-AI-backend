"""Run listing and fetch endpoints, scoped to the authenticated user.

Each user's runs live under data/runs/{user_id}/ — the listing globs only
that subdir, the detail endpoint refuses to read across users. In dev-local
mode the user_id is `dev-local` and everything still resolves to a single
folder, so a single dev keeps seeing all their work.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from auth import CurrentUser
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


@router.get("/runs")
def list_runs(user: CurrentUser) -> list[dict[str, Any]]:
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

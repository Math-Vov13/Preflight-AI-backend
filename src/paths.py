"""Shared path anchors — everything resolves from the repo root so scripts and
the FastAPI server write to the same `data/runs/` directory regardless of CWD.
"""
from __future__ import annotations

import re
from pathlib import Path

# backend/app/paths.py → parents[2] = repo root
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data"
RUNS_DIR: Path = DATA_DIR / "runs"


_SAFE_RE = re.compile(r"[^a-zA-Z0-9_\-]")


def user_runs_dir(user_id: str) -> Path:
    """Return the per-user run directory under data/runs/{user_id}/.

    user_ids come from Supabase (UUIDs) or the dev-local fallback string.
    We sanitize anything that isn't already filesystem-safe so a malformed
    JWT can't traverse out of data/runs/. The mkdir-on-read happens at the
    caller — this helper only computes the path.
    """
    safe = _SAFE_RE.sub("_", user_id) if user_id else "anon"
    return RUNS_DIR / safe

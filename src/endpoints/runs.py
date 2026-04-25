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

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from auth import CurrentUser
from models import preflight_db
from paths import user_runs_dir

router = APIRouter(tags=["runs"])

# File naming convention inherited from ai-agent-demo; may rename to "run_" later.
_FILE_PREFIX = "pre_demo_"

# Mirrors the run_status pgEnum (Drizzle-side runs.sql.ts). Kept in sync
# manually — tests would catch a drift.
_RUN_STATUSES: frozenset[str] = frozenset({"running", "done", "error"})

# Mirrors the artefact_kind pgEnum. Same drift-protection contract.
_ARTEFACT_KINDS: frozenset[str] = frozenset({
    "ontology", "panel", "thread", "validation_report", "judge_scores",
})


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


def _iso(ts: Any) -> str | None:
    """Postgres timestamps come back as datetime; serialize to ISO so JSON
    encoding succeeds and the frontend can `new Date(s)` directly."""
    return ts.isoformat() if ts is not None else None


def _summary_from_db_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project a runs row into the listing shape the frontend expects.

    `timestamp` is sourced from `started_at` now that run_ids are UUIDs and
    no longer self-describing. `status` + `error_message` let the sidebar
    distinguish in-flight / errored / done runs without an extra request.
    `title` (when set via PATCH /runs/{id}) lets the sidebar show a
    user-chosen label instead of the brief preview.
    """
    started_iso = _iso(row.get("started_at"))
    settings = row.get("settings") or {}
    title = settings.get("title") if isinstance(settings, dict) else None
    return {
        "id": row["id"],
        "timestamp": started_iso or row["id"],
        "started_at": started_iso,
        "completed_at": _iso(row.get("completed_at")),
        "status": row.get("status"),
        "error_message": row.get("error_message"),
        "title": title,
        "brief_preview": (row.get("brief") or "")[:300],
        "panel_size": row.get("panel_size"),
        "rounds": row.get("rounds"),
        "total_latency_s": float(row["wall_s"]) if row.get("wall_s") is not None else None,
        "total_cost_usd": float(row["cost_usd"]) if row.get("cost_usd") is not None else None,
        "go_no_go_recommendation": row.get("verdict"),
    }


def _empty_stats() -> dict[str, Any]:
    return {
        "total": 0,
        "by_status": {"running": 0, "done": 0, "error": 0},
        "by_verdict": {"go": 0, "pivot": 0, "kill": 0, "unknown": 0},
        "total_cost_usd": 0.0,
        "first_run_at": None,
        "last_run_at": None,
    }


@router.get("/runs/me/stats")
def my_run_stats(user: CurrentUser) -> dict[str, Any]:
    """Dashboard aggregate for the authenticated user. Identical shape
    in DB-mode and file-mode; file-mode reads each metrics sidecar to
    rebuild the verdict/cost rollups."""
    db_stats = preflight_db.user_run_stats(user)
    if db_stats is not None:
        return db_stats

    # ── File-mode fallback ───────────────────────────────────────────
    runs_dir = user_runs_dir(user)
    if not runs_dir.exists():
        return _empty_stats()
    out = _empty_stats()
    timestamps: list[str] = []
    for art_path in sorted(runs_dir.glob(f"{_FILE_PREFIX}*.json")):
        if art_path.name.endswith(("_metrics.json", "_chat.json")):
            continue
        run_id = _run_id(art_path)
        timestamps.append(run_id)
        out["total"] += 1
        metrics = _load_json(art_path.with_name(f"{art_path.stem}_metrics.json")) or {}
        run_block = metrics.get("run", {}) or {}
        cost_block = metrics.get("cost", {}) or {}
        verdict = run_block.get("go_no_go_recommendation")
        # File-mode runs are always terminal (artefacts only land after
        # persist_run). 'done' if a verdict survived, 'error' otherwise.
        if verdict:
            out["by_status"]["done"] += 1
            bucket = verdict if verdict in out["by_verdict"] else "unknown"
            out["by_verdict"][bucket] += 1
        else:
            out["by_status"]["error"] += 1
            out["by_verdict"]["unknown"] += 1
        cost = cost_block.get("total_usd") or 0
        try:
            out["total_cost_usd"] += float(cost)
        except (TypeError, ValueError):
            pass
    if timestamps:
        out["first_run_at"] = min(timestamps)
        out["last_run_at"] = max(timestamps)
    out["total_cost_usd"] = round(out["total_cost_usd"], 6)
    return out


@router.get("/runs")
def list_runs(
    user: CurrentUser,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    status: str | None = Query(default=None),
) -> list[dict[str, Any]]:
    if status is not None and status not in _RUN_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status; expected one of {sorted(_RUN_STATUSES)}",
        )

    # Try DB first. None means "DB unavailable for this user" — fall back
    # to file scan. Empty list means "DB available, no runs yet" — return
    # it as-is so we don't double-list anything that's also on disk.
    db_rows = preflight_db.list_runs_for_user(
        user, limit=limit, offset=offset, status=status,
    )
    if db_rows is not None:
        return [_summary_from_db_row(r) for r in db_rows]

    # ── File-mode fallback ───────────────────────────────────────────
    # File-mode runs are always terminal (artefacts are only written by
    # persist_run, which runs after the pipeline completes). So filtering
    # by status='running' yields nothing; status='error'/'done' need a
    # peek at the metrics block.
    if status == "running":
        return []
    runs_dir = user_runs_dir(user)
    if not runs_dir.exists():
        return []
    primary_paths = [
        p for p in sorted(runs_dir.glob(f"{_FILE_PREFIX}*.json"), reverse=True)
        if not p.name.endswith(("_metrics.json", "_chat.json"))
    ]
    out: list[dict[str, Any]] = []
    for art_path in primary_paths:
        run_id = _run_id(art_path)
        metrics = _load_json(art_path.with_name(f"{art_path.stem}_metrics.json")) or {}
        run_block = metrics.get("run", {}) or {}
        cost_block = metrics.get("cost", {}) or {}
        # File-mode pre-dates the status field; treat presence of a
        # validation verdict as "done" and infer "error" from a recorded
        # error block. Anything else passes the filter unfiltered.
        inferred_status = "done" if run_block.get("go_no_go_recommendation") else None
        if status is not None and inferred_status != status:
            continue
        out.append(
            {
                "id": run_id,
                "timestamp": run_id,
                **run_block,
                "total_cost_usd": cost_block.get("total_usd"),
                "calls": cost_block.get("calls"),
            }
        )
        if len(out) >= limit + offset:
            break
    return out[offset:offset + limit]


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
        started_iso = _iso(db_run.get("started_at"))
        return {
            "id": run_id,
            "timestamp": started_iso or run_id,
            "started_at": started_iso,
            "completed_at": _iso(db_run.get("completed_at")),
            "status": db_run.get("status"),
            "error_message": db_run.get("error_message"),
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


def _render_transcript(run: dict[str, Any]) -> str:
    """Compose a markdown export from the same shape get_run returns.

    Defensive against missing pieces — runs that errored out before
    validation finished still produce a usable transcript with
    'Validation incomplete' placeholders rather than blowing up.
    """
    artefacts = run.get("artefacts") or {}
    brief = (artefacts.get("brief") or "").strip()
    title = run.get("title") or "PreFlight Run"
    started = run.get("started_at") or run.get("timestamp") or ""
    status = run.get("status") or "unknown"
    metrics_run = (run.get("metrics") or {}).get("run") or {}
    cost_block = (run.get("metrics") or {}).get("cost") or {}

    lines: list[str] = [
        f"# {title}",
        "",
        f"- **Run ID:** `{run.get('id', '')}`",
        f"- **Status:** {status}",
        f"- **Started:** {started}",
    ]
    if metrics_run.get("panel_size"):
        lines.append(
            f"- **Panel:** {metrics_run.get('panel_size')} personas, "
            f"{metrics_run.get('rounds')} rounds",
        )
    if cost_block.get("total_usd") is not None:
        lines.append(f"- **Cost:** ${float(cost_block['total_usd']):.4f}")
    lines.append("")

    if brief:
        lines.extend(["## Brief", "", brief, ""])

    report = artefacts.get("validation_report")
    if not isinstance(report, dict):
        lines.extend(["## Validation", "", "_Validation incomplete._", ""])
        return "\n".join(lines)

    verdict = report.get("go_no_go_recommendation") or "—"
    rationale = (report.get("go_no_go_rationale") or "").strip()
    lines.extend([
        "## Verdict",
        "",
        f"**{verdict.upper()}**",
        "",
    ])
    if rationale:
        lines.extend([rationale, ""])

    panel_comp = report.get("panel_composition") or {}
    if panel_comp:
        lines.append("## Panel composition")
        lines.append("")
        for seg, n in panel_comp.items():
            lines.append(f"- {seg}: {n}")
        lines.append("")

    objections = report.get("top_objections") or []
    if objections:
        lines.append("## Top objections")
        lines.append("")
        for o in objections:
            text = (o.get("text") or "").strip()
            severity = o.get("severity") or "?"
            freq = o.get("frequency", 0)
            lines.append(f"- **[{severity}]** {text} _(×{freq})_")
        lines.append("")

    missing = report.get("missing_features") or []
    if missing:
        lines.append("## Missing features")
        lines.append("")
        for m in missing:
            feat = (m.get("feature") or "").strip()
            n = m.get("requested_by_n", 0)
            lines.append(f"- {feat} _(requested by {n})_")
        lines.append("")

    pricing = report.get("pricing_feedback")
    if isinstance(pricing, dict):
        floor = pricing.get("floor_eur_month")
        ceiling = pricing.get("ceiling_eur_month")
        if floor or ceiling:
            lines.append("## Pricing feedback")
            lines.append("")
            lines.append(f"- Floor: €{floor or 0:.0f}/mo")
            lines.append(f"- Ceiling: €{ceiling or 0:.0f}/mo")
            lines.append(
                f"- Would pay announced price: "
                f"{pricing.get('n_would_pay_announced_price', 0)} / "
                f"refused: {pricing.get('n_would_not_pay_announced_price', 0)}",
            )
            lines.append("")

    red_flags = report.get("red_flags") or []
    if red_flags:
        lines.append("## Red flags")
        lines.append("")
        for f in red_flags:
            lines.append(f"- {f}")
        lines.append("")

    cuts = report.get("recommended_mvp_cuts") or []
    if cuts:
        lines.append("## Recommended MVP cuts")
        lines.append("")
        for c in cuts:
            lines.append(f"- {c}")
        lines.append("")

    return "\n".join(lines)


@router.get("/runs/{run_id}/transcript", response_class=PlainTextResponse)
def get_run_transcript(run_id: str, user: CurrentUser) -> PlainTextResponse:
    """Markdown export of a run for sharing / copy-paste / download.

    Reuses the same projection get_run returns so DB-mode and file-mode
    transcripts come out identical in shape (frontend can deep-link to
    a `.md` download regardless of where the run lives).
    """
    # Lean on the existing endpoint's data composition — calling the
    # function directly avoids re-implementing the DB/file branching.
    full = get_run(run_id, user)  # raises 404 itself when missing
    body = _render_transcript(full)
    return PlainTextResponse(
        body,
        media_type="text/markdown; charset=utf-8",
        headers={
            # Encourage browsers to download rather than render inline.
            "Content-Disposition": f'attachment; filename="run-{run_id}.md"',
        },
    )


def _file_mode_artefact(user_id: str, run_id: str, kind: str) -> dict[str, Any] | None:
    """File-mode equivalent of preflight_db.get_artefact — slices the
    primary artefacts.json to one kind. Returns the wrapped envelope or
    None if the run / kind isn't present."""
    art_path = user_runs_dir(user_id) / f"{_FILE_PREFIX}{run_id}.json"
    if not art_path.exists():
        return None
    artefacts = _load_json(art_path) or {}
    if kind == "judge_scores":
        # Judge scores live in the metrics sidecar, not artefacts.json,
        # because they're a quality signal *about* the artefacts.
        metrics = _load_json(art_path.with_name(f"{art_path.stem}_metrics.json")) or {}
        payload = metrics.get("judge")
    elif kind == "panel":
        # File-mode stores panel as a list directly; wrap to match
        # DB-mode shape ({"personas": [...]}).
        personas = artefacts.get("panel")
        payload = {"personas": personas} if personas is not None else None
    else:
        payload = artefacts.get(kind)
    if payload is None:
        return None
    return {
        "kind": kind,
        "payload": payload,
        "created_at": None,
        "updated_at": None,
    }


@router.get("/runs/{run_id}/artefacts/{kind}")
def get_run_artefact(
    run_id: str, kind: str, user: CurrentUser,
) -> dict[str, Any]:
    """Fetch a single artefact for lazy-loading. Avoids the cost of
    pulling the full /runs/{id} payload when the frontend only needs
    one section (e.g. just the validation_report tab)."""
    if kind not in _ARTEFACT_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid artefact kind; expected one of {sorted(_ARTEFACT_KINDS)}",
        )
    db_artefact = preflight_db.get_artefact(
        run_id=run_id, kind=kind, auth_uid=user,
    )
    if db_artefact is not None:
        return db_artefact

    # ── File-mode fallback ───────────────────────────────────────────
    file_artefact = _file_mode_artefact(user, run_id, kind)
    if file_artefact is not None:
        return file_artefact
    raise HTTPException(
        status_code=404,
        detail=f"Artefact '{kind}' not found for run {run_id}",
    )


def _delete_file_artefacts(user_id: str, run_id: str) -> int:
    """Wipe every pre_demo_{run_id}*.json|.parquet sidecar. Returns the
    count of files removed. Used by the DELETE endpoint's file-mode path
    *and* as a belt-and-braces cleanup after a DB delete."""
    runs_dir = user_runs_dir(user_id)
    if not runs_dir.exists():
        return 0
    removed = 0
    for p in runs_dir.glob(f"{_FILE_PREFIX}{run_id}*"):
        try:
            p.unlink()
            removed += 1
        except OSError:
            # Best-effort; don't fail the request because a sidecar was
            # write-locked or already gone.
            pass
    return removed


class PatchRunRequest(BaseModel):
    title: str | None = Field(default=None, max_length=120)


@router.patch("/runs/{run_id}")
def patch_run(
    run_id: str, body: PatchRunRequest, user: CurrentUser,
) -> dict[str, Any]:
    """Update mutable run metadata. Currently just `title` (stored in
    settings.title). Returns the updated summary projection so the
    frontend can replace its sidebar entry without a refetch."""
    patch: dict[str, Any] = {}
    if body.title is not None:
        # Empty string is a valid 'unset' — clears the title and
        # the sidebar falls back to brief preview.
        patch["title"] = body.title.strip()
    if not patch:
        raise HTTPException(status_code=400, detail="No fields to update")

    result = preflight_db.update_run_settings(
        run_id=run_id, auth_uid=user, patch=patch,
    )
    if result is None:
        # DB-unavailable / dev-local — patching settings only makes sense
        # against the durable store; file-mode artefacts don't have a
        # settings concept. 503 so the frontend knows it should retry
        # once the DB is back.
        raise HTTPException(
            status_code=503,
            detail="Run settings can only be updated when DB persistence is enabled",
        )
    if not result:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    # Re-fetch the row so the response reflects merged state.
    db_run = preflight_db.get_run_with_artifacts(run_id=run_id, auth_uid=user)
    if db_run is None:  # pragma: no cover — race with DELETE
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    db_run["id"] = run_id
    return _summary_from_db_row(db_run)


@router.delete("/runs/{run_id}", status_code=204)
def delete_run(run_id: str, user: CurrentUser) -> None:
    """Hard-delete a run + every sidecar/artefact owned by this user.

    DB rows cascade via the run_artifacts FK. File-mode sidecars are
    removed in the same call so the listing reflects the delete in both
    modes. In-flight runs are refused — clients should POST /cancel
    first.
    """
    db_result = preflight_db.delete_run(run_id=run_id, auth_uid=user)
    if db_result == "running":
        raise HTTPException(
            status_code=409,
            detail="Run is in progress — POST /runs/{id}/cancel first.",
        )
    if db_result == "deleted":
        # Belt-and-braces: wipe any leftover on-disk sidecars from before
        # BE-PR1 too, so a hybrid run doesn't leave orphans.
        _delete_file_artefacts(user, run_id)
        return None
    # db_result is "not_found" or None (DB unavailable). Fall through
    # to the file-mode path: a missing file is the only way to 404.
    removed = _delete_file_artefacts(user, run_id)
    if removed == 0 and db_result == "not_found":
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return None

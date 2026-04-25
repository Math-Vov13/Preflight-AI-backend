"""Projection shape tests for endpoints/runs.py.

The frontend lib/types/preflight.ts mirrors these shapes by hand. A
test here catches drift between what the backend serialises and what
the frontend parses.
"""
from __future__ import annotations

from datetime import datetime

from endpoints import runs as runs_endpoint


def test_summary_from_db_row_iso_serialises_timestamps() -> None:
    row = {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "brief": "Some brief",
        "panel_size": 10,
        "rounds": 3,
        "status": "running",
        "verdict": None,
        "cost_usd": None,
        "wall_s": None,
        "rationale": None,
        "error_message": None,
        "started_at": datetime(2026, 4, 26, 9, 30),
        "completed_at": None,
    }
    out = runs_endpoint._summary_from_db_row(row)  # noqa: SLF001
    assert out["id"] == row["id"]
    assert out["status"] == "running"
    # timestamp falls back to started_at when present (UUID ids aren't
    # human-readable; the frontend treats this field as a date).
    assert out["timestamp"] == "2026-04-26T09:30:00"
    assert out["started_at"] == "2026-04-26T09:30:00"
    assert out["completed_at"] is None
    assert out["brief_preview"] == "Some brief"


def test_summary_truncates_long_brief() -> None:
    row = {
        "id": "x",
        "brief": "a" * 1000,
        "started_at": datetime(2026, 4, 26),
    }
    out = runs_endpoint._summary_from_db_row(row)  # noqa: SLF001
    assert len(out["brief_preview"]) == 300


def test_summary_handles_decimal_typed_costs() -> None:
    """psycopg returns numeric columns as Decimal; we coerce to float
    for JSON serialisability."""
    from decimal import Decimal

    row = {
        "id": "x",
        "brief": "",
        "wall_s": Decimal("12.34"),
        "cost_usd": Decimal("0.4567"),
        "started_at": datetime(2026, 4, 26),
    }
    out = runs_endpoint._summary_from_db_row(row)  # noqa: SLF001
    assert out["total_latency_s"] == 12.34
    assert out["total_cost_usd"] == 0.4567


def test_run_status_constant_matches_drizzle_enum() -> None:
    """If this test fails, somebody added a value to the run_status
    pgEnum on the frontend without updating the backend filter
    whitelist."""
    assert runs_endpoint._RUN_STATUSES == frozenset({"running", "done", "error"})  # noqa: SLF001


def test_artefact_kinds_constant_matches_drizzle_enum() -> None:
    """Same drift-protection, for artefact_kind."""
    assert runs_endpoint._ARTEFACT_KINDS == frozenset({  # noqa: SLF001
        "ontology", "panel", "thread", "validation_report", "judge_scores",
    })


def test_summary_picks_up_title_from_settings() -> None:
    """BE-PR19: settings.title overrides the default brief-preview label."""
    row = {
        "id": "x",
        "brief": "long brief about a product idea",
        "settings": {"title": "Q3 Marketing Test", "simulation_seed": 42},
        "started_at": datetime(2026, 4, 26),
    }
    out = runs_endpoint._summary_from_db_row(row)  # noqa: SLF001
    assert out["title"] == "Q3 Marketing Test"


def test_summary_title_is_none_when_settings_omits_it() -> None:
    row = {
        "id": "x",
        "brief": "x",
        "settings": {"simulation_seed": 42},
        "started_at": datetime(2026, 4, 26),
    }
    out = runs_endpoint._summary_from_db_row(row)  # noqa: SLF001
    assert out["title"] is None


def test_summary_title_is_none_when_settings_is_null() -> None:
    """Defensive — old runs created before BE-PR1 might lack settings."""
    row = {
        "id": "x",
        "brief": "x",
        "settings": None,
        "started_at": datetime(2026, 4, 26),
    }
    out = runs_endpoint._summary_from_db_row(row)  # noqa: SLF001
    assert out["title"] is None


def test_transcript_renders_minimal_run() -> None:
    """A run with brief but no validation report still renders."""
    md = runs_endpoint._render_transcript({  # noqa: SLF001
        "id": "abc",
        "title": None,
        "started_at": "2026-04-26T01:00:00",
        "status": "error",
        "metrics": {"run": {}},
        "artefacts": {"brief": "Test brief"},
    })
    assert md.startswith("# PreFlight Run")
    assert "## Brief" in md
    assert "Test brief" in md
    assert "_Validation incomplete._" in md
    assert "## Verdict" not in md  # report missing → no verdict block


def test_transcript_renders_full_report() -> None:
    md = runs_endpoint._render_transcript({  # noqa: SLF001
        "id": "abc",
        "title": "Q3 Marketing Test",
        "started_at": "2026-04-26T01:00:00",
        "status": "done",
        "metrics": {
            "run": {"panel_size": 10, "rounds": 3},
            "cost": {"total_usd": 1.2345},
        },
        "artefacts": {
            "brief": "Hello",
            "validation_report": {
                "go_no_go_recommendation": "go",
                "go_no_go_rationale": "Strong validation across segments.",
                "panel_composition": {"freelancers": 6, "founders": 4},
                "top_objections": [
                    {"text": "Too expensive", "severity": "blocker", "frequency": 3},
                ],
                "missing_features": [
                    {"feature": "calendar sync", "requested_by_n": 5},
                ],
                "pricing_feedback": {
                    "floor_eur_month": 9.0,
                    "ceiling_eur_month": 29.0,
                    "n_would_pay_announced_price": 6,
                    "n_would_not_pay_announced_price": 4,
                },
                "red_flags": ["legal risk in EU"],
                "recommended_mvp_cuts": ["video calling"],
            },
        },
    })
    # Header + metadata
    assert "# Q3 Marketing Test" in md
    assert "**Status:** done" in md
    assert "$1.2345" in md
    # Verdict block
    assert "**GO**" in md
    assert "Strong validation across segments." in md
    # Sections
    assert "## Top objections" in md
    assert "**[blocker]** Too expensive _(×3)_" in md
    assert "## Missing features" in md
    assert "calendar sync _(requested by 5)_" in md
    assert "## Pricing feedback" in md
    assert "Floor: €9/mo" in md
    assert "## Red flags" in md
    assert "## Recommended MVP cuts" in md

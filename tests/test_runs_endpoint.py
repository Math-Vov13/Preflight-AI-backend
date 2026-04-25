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

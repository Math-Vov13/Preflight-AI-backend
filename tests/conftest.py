"""Pytest fixtures shared across the preflight test suite.

`pythonpath = ["src"]` in pyproject already exposes the `src/` modules to
imports, but Python loads `.env` only when the application boots. Tests
that touch settings need a clean environment so they don't accidentally
hit prod (Postgres, Zep, Supabase).
"""
from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Force every test to run as if no external services are configured.

    Tests that DO want a feature can re-set the var with monkeypatch
    inside the test body. Default-deny here keeps a stray `import server`
    from connecting to the real Qdrant / Postgres clusters.
    """
    for var in (
        "DATABASE_URL",
        "ZEP_API_KEY",
        "SUPABASE_JWT_SECRET",
        "QDRANT_URL",
        "QDRANT_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DEV_USER_ID", "test-user")
    yield

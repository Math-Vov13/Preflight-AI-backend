"""Per-run model_overrides validation (BE-PR25).

Wires:
  StartRunRequest.model_overrides (dict)
   → control._OVERRIDABLE_PHASES whitelist
   → run_full_pipeline(model_overrides=...)
   → each service's `model=...` constructor arg

These tests pin the whitelist contract. Full pipeline integration is
out of scope (would need a live LLM); we only verify the validation
layer + the constants the endpoint and pipeline both use.
"""
from __future__ import annotations

import inspect

import pytest
from fastapi import HTTPException

from endpoints import control
from services.pipeline import run_full_pipeline


def test_overridable_phases_constant() -> None:
    """Drift catcher: pipeline reads overrides via the same key set."""
    assert control._OVERRIDABLE_PHASES == frozenset({  # noqa: SLF001
        "ontology", "persona", "simulation", "report", "judge",
    })


def test_pipeline_accepts_model_overrides_kwarg() -> None:
    """run_full_pipeline must expose model_overrides — without it the
    endpoint's plumbing wouldn't reach the services."""
    sig = inspect.signature(run_full_pipeline)
    assert "model_overrides" in sig.parameters
    param = sig.parameters["model_overrides"]
    assert param.default is None  # optional, not required


async def test_start_run_rejects_unknown_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bad keys 400 before claiming any lock or DB row."""
    from endpoints.control import StartRunRequest, start_run

    req = StartRunRequest(
        brief="x", panel_size=5, rounds=2,
        model_overrides={"ontology": "foo/bar", "bogus": "x"},
    )
    with pytest.raises(HTTPException) as exc:
        await start_run(req, user="dev-local", idempotency_key=None)
    assert exc.value.status_code == 400
    assert "bogus" in exc.value.detail

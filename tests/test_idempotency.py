"""Unit tests for the in-process idempotency cache used by /runs/new
(BE-PR15). Covers TTL expiry, per-user namespacing, and the no-key
no-op."""
from __future__ import annotations

import time

import pytest

from endpoints import control


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    control._idempotency_cache.clear()  # noqa: SLF001


def test_no_key_is_noop() -> None:
    assert control._idempotency_lookup("user-A", None) is None  # noqa: SLF001
    control._idempotency_remember("user-A", None, "run-1")  # noqa: SLF001
    assert control._idempotency_cache == {}  # noqa: SLF001


def test_remember_then_lookup_returns_run_id() -> None:
    control._idempotency_remember("user-A", "key-1", "run-1")  # noqa: SLF001
    assert control._idempotency_lookup("user-A", "key-1") == "run-1"  # noqa: SLF001


def test_keys_are_per_user_namespaced() -> None:
    control._idempotency_remember("user-A", "key-1", "run-1")  # noqa: SLF001
    control._idempotency_remember("user-B", "key-1", "run-2")  # noqa: SLF001
    assert control._idempotency_lookup("user-A", "key-1") == "run-1"  # noqa: SLF001
    assert control._idempotency_lookup("user-B", "key-1") == "run-2"  # noqa: SLF001


def test_expired_entries_are_evicted_on_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lookup walks the dict and GCs expired keys — a long-running
    server doesn't accumulate unbounded state."""
    control._idempotency_remember("user-A", "key-1", "run-1")  # noqa: SLF001
    # Patch *control's* view of time, not the time module — otherwise
    # the lambda recurses through its own patched binding.
    fake_now = time.time() + control._IDEMPOTENCY_TTL_S + 1  # noqa: SLF001
    monkeypatch.setattr(control.time, "time", lambda: fake_now)
    assert control._idempotency_lookup("user-A", "key-1") is None  # noqa: SLF001
    assert ("user-A", "key-1") not in control._idempotency_cache  # noqa: SLF001

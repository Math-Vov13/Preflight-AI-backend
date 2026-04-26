"""TokenBucket unit tests (BE-PR16)."""
from __future__ import annotations

import time

import pytest

from rate_limit import TokenBucket


def test_first_consume_succeeds() -> None:
    bucket = TokenBucket(capacity=3, refill_per_sec=1.0)
    assert bucket.try_consume("user-A") is None


def test_burst_then_reject() -> None:
    bucket = TokenBucket(capacity=3, refill_per_sec=0.001)  # ~slow refill
    for _ in range(3):
        assert bucket.try_consume("user-A") is None
    retry_after = bucket.try_consume("user-A")
    assert retry_after is not None
    assert retry_after >= 1  # always rounds up to ≥1 second


def test_per_user_isolation() -> None:
    bucket = TokenBucket(capacity=1, refill_per_sec=0.001)
    assert bucket.try_consume("user-A") is None
    # User A is now empty; user B still has full capacity.
    assert bucket.try_consume("user-B") is None
    assert bucket.try_consume("user-A") is not None


def test_refill_replenishes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refill formula: tokens += elapsed * refill_per_sec, capped at capacity."""
    import rate_limit as rl_module

    fake_now = 1000.0
    monkeypatch.setattr(rl_module.time, "monotonic", lambda: fake_now)

    bucket = TokenBucket(capacity=2, refill_per_sec=1.0)
    assert bucket.try_consume("u") is None  # 2 -> 1
    assert bucket.try_consume("u") is None  # 1 -> 0
    assert bucket.try_consume("u") is not None  # rejected

    # 2 seconds later the bucket is full again
    fake_now += 2.0
    assert bucket.try_consume("u") is None  # 2 -> 1
    assert bucket.try_consume("u") is None  # 1 -> 0


def test_reset_clears_state() -> None:
    bucket = TokenBucket(capacity=1, refill_per_sec=0.001)
    bucket.try_consume("u")
    assert bucket.try_consume("u") is not None
    bucket.reset("u")
    assert bucket.try_consume("u") is None

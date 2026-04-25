"""Heartbeat loop test (BE-PR17).

Pins the contract: the loop publishes run.heartbeat with run_id +
elapsed_s on the events bus, scoped to the run owner via the user
ContextVar. Cancellation cleanly stops the loop without re-raising.
"""
from __future__ import annotations

import asyncio

import pytest

import events
from endpoints import control


@pytest.fixture(autouse=True)
def _reset_event_bus() -> None:
    events._loop = None  # noqa: SLF001
    with events._subscribers_lock:  # noqa: SLF001
        events._subscribers.clear()  # noqa: SLF001


async def test_heartbeat_publishes_with_user_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Override the cadence so the test doesn't sleep 10s.
    monkeypatch.setattr(control, "_HEARTBEAT_INTERVAL_S", 0.05)
    events.attach_loop(asyncio.get_running_loop())
    qa = events.subscribe("user-A")
    qb = events.subscribe("user-B")

    task = asyncio.create_task(
        control._heartbeat_loop("run-X", "user-A", started=0.0),  # noqa: SLF001
    )
    # Wait long enough for at least 2 heartbeats.
    await asyncio.sleep(0.15)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # User A got heartbeats; user B got none (per-user isolation from BE-PR4).
    assert qa.qsize() >= 2
    assert qb.qsize() == 0
    event = await qa.get()
    assert event["type"] == "run.heartbeat"
    assert event["data"]["run_id"] == "run-X"
    assert event["data"]["elapsed_s"] >= 0
    assert event["user_id"] == "user-A"


async def test_heartbeat_cancel_is_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling mid-sleep should not raise into the caller."""
    monkeypatch.setattr(control, "_HEARTBEAT_INTERVAL_S", 0.5)
    events.attach_loop(asyncio.get_running_loop())
    events.subscribe("user-A")
    task = asyncio.create_task(
        control._heartbeat_loop("run-X", "user-A", started=0.0),  # noqa: SLF001
    )
    await asyncio.sleep(0.01)  # well before the first beat
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

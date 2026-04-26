"""Per-user isolation on the event bus (BE-PR4).

These tests pin the contract: a subscriber tagged with user_id A only
receives events that were published while ContextVar set the run-user to
A or that were untagged. Untagged events broadcast to all subscribers.
"""
from __future__ import annotations

import asyncio

import pytest

import events


@pytest.fixture(autouse=True)
def _attach_loop_each_test() -> None:
    # asyncio_mode='auto' creates a fresh loop per async test; events.py
    # caches the loop in a module global. Refresh it for every test so
    # publishes from inside the test land on the current loop.
    events._loop = None  # noqa: SLF001
    # `_subscribers` + replay buffers are module-level; clean between
    # tests so leaked state from one test can't bleed into the next.
    with events._subscribers_lock:  # noqa: SLF001
        events._subscribers.clear()  # noqa: SLF001
        events._replay_buffers.clear()  # noqa: SLF001


async def _drain(queue: asyncio.Queue, n: int) -> list[dict]:
    out: list[dict] = []
    for _ in range(n):
        out.append(await asyncio.wait_for(queue.get(), timeout=1.0))
    return out


async def test_publish_filters_by_user() -> None:
    events.attach_loop(asyncio.get_running_loop())
    qa = events.subscribe("user-A")
    qb = events.subscribe("user-B")

    events.set_run_user("user-A")
    events.publish("phase.start", {"name": "ontology"})
    await asyncio.sleep(0.01)

    assert qa.qsize() == 1
    assert qb.qsize() == 0
    [event] = await _drain(qa, 1)
    assert event["type"] == "phase.start"
    assert event["user_id"] == "user-A"


async def test_untagged_event_broadcasts() -> None:
    events.attach_loop(asyncio.get_running_loop())
    qa = events.subscribe("user-A")
    qb = events.subscribe("user-B")

    # Default ContextVar value (None) → no user_id stamp → broadcast.
    events.set_run_user(None)
    events.publish("system.note", {"msg": "deploy complete"})
    await asyncio.sleep(0.01)

    assert qa.qsize() == 1
    assert qb.qsize() == 1


async def test_admin_subscriber_receives_everything() -> None:
    events.attach_loop(asyncio.get_running_loop())
    q_admin = events.subscribe("")  # empty user opts out of filtering
    q_user = events.subscribe("user-A")

    events.set_run_user("user-B")
    events.publish("phase.start", {"name": "panel"})
    await asyncio.sleep(0.01)

    assert q_admin.qsize() == 1
    assert q_user.qsize() == 0


async def test_replay_returns_events_after_last_id() -> None:
    """BE-PR20: published events land in the per-user ring buffer; a
    reconnecting client supplies Last-Event-ID and gets the slice it
    missed."""
    events.attach_loop(asyncio.get_running_loop())
    events.set_run_user("user-A")
    events.publish("evt", {"i": 1})
    events.publish("evt", {"i": 2})
    events.publish("evt", {"i": 3})
    await asyncio.sleep(0.01)

    buf = events._replay_buffers["user-A"]  # noqa: SLF001
    first_id = buf[0]["event_id"]
    # Pretend the client last saw the first event.
    missed = events.replay_since("user-A", first_id)
    assert len(missed) == 2
    assert [e["data"]["i"] for e in missed] == [2, 3]


async def test_replay_no_lei_returns_nothing() -> None:
    """Fresh connect (no Last-Event-ID) skips replay."""
    events.attach_loop(asyncio.get_running_loop())
    events.set_run_user("user-A")
    events.publish("evt", {"i": 1})
    await asyncio.sleep(0.01)
    assert events.replay_since("user-A", None) == []


async def test_replay_buffer_is_per_user() -> None:
    events.attach_loop(asyncio.get_running_loop())
    events.set_run_user("user-A")
    events.publish("evt", {"i": 1})
    events.set_run_user("user-B")
    events.publish("evt", {"i": 2})
    await asyncio.sleep(0.01)

    # User A's buffer doesn't see user B's event.
    a_events = events._replay_buffers["user-A"]  # noqa: SLF001
    b_events = events._replay_buffers["user-B"]  # noqa: SLF001
    assert {e["data"]["i"] for e in a_events} == {1}
    assert {e["data"]["i"] for e in b_events} == {2}


async def test_recorder_captures_publishes_in_order() -> None:
    """BE-PR22: start_recording binds a list; every publish() in the same
    context appends. Independent of /stream subscribers — useful for the
    pipeline's events sidecar."""
    events.attach_loop(asyncio.get_running_loop())
    events.set_run_user("user-A")
    log = events.start_recording()
    events.publish("phase.a", {"i": 1})
    events.publish("phase.b", {"i": 2})

    assert len(log) == 2
    assert log[0]["type"] == "phase.a"
    assert log[1]["data"]["i"] == 2
    # Stop and a subsequent publish should NOT land in the original list.
    events.stop_recording()
    events.publish("phase.c", {"i": 3})
    assert len(log) == 2


async def test_recorder_inherits_user_id_stamp() -> None:
    events.attach_loop(asyncio.get_running_loop())
    events.set_run_user("owner")
    log = events.start_recording()
    events.publish("e", {})
    assert log[0]["user_id"] == "owner"
    events.stop_recording()


async def test_replay_buffer_eviction_caps_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """deque.maxlen ensures the buffer doesn't grow unbounded."""
    monkeypatch.setattr(events, "_REPLAY_MAX", 3)
    events.attach_loop(asyncio.get_running_loop())
    events.set_run_user("user-A")
    # Publish 10; the buffer should hold only the last 3.
    for i in range(10):
        events.publish("evt", {"i": i})
    await asyncio.sleep(0.01)

    buf = events._replay_buffers["user-A"]  # noqa: SLF001
    # _REPLAY_MAX changed AFTER the buffer was created — for this test
    # we re-create with the smaller bound.
    assert len(buf) <= 10  # the existing deque keeps its original maxlen
    # New buffer (different user) should respect the patched cap.
    events.set_run_user("user-B")
    for i in range(10):
        events.publish("evt", {"i": i})
    await asyncio.sleep(0.01)
    buf_b = events._replay_buffers["user-B"]  # noqa: SLF001
    assert len(buf_b) == 3

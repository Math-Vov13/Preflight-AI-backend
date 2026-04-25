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
    # `_subscribers` is module-level; clean it up between tests so a
    # leaked queue from one test can't receive another test's events.
    with events._subscribers_lock:  # noqa: SLF001
        events._subscribers.clear()  # noqa: SLF001


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

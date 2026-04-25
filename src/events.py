"""Thread-safe pub/sub event bus for runtime observability.

Runner threads call `publish(...)` — fire-and-forget, safe from any thread.
The SSE endpoint calls `subscribe(user_id)` from the async event loop and
drains per-subscriber queues. Events never block the runner; queue
overflow drops the oldest event.

Per-user event isolation
------------------------
`run_full_pipeline` (and any code path that wants its publishes scoped to
a particular user) sets the `_user_var` ContextVar at entry. `publish()`
reads that var and stamps `user_id` onto the event. The dispatcher then
only enqueues an event for a subscriber whose `user_id` matches — or for
all subscribers when the event has no user_id (legacy / system events).

ContextVar propagates across `asyncio.to_thread` automatically. For
ThreadPoolExecutor work spawned inside the pipeline (see
persona_generator), pass `contextvars.copy_context().run` so the children
inherit too.
"""
from __future__ import annotations

import asyncio
import contextvars
import threading
import time
from typing import Any

Event = dict[str, Any]

# Subscriber tuple = (queue, user_id). user_id "" means "no scoping",
# receives every event. The stream endpoint always passes a real user_id;
# the empty-string case is reserved for tests and CLI consumers.
_subscribers: list[tuple[asyncio.Queue[Event], str]] = []
_subscribers_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None

_QUEUE_MAX = 5000

# Set at the top of a pipeline run so every downstream publish() inherits
# the owning user. None means "no scoping → broadcast".
_user_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "preflight_event_user", default=None,
)


def attach_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Register the FastAPI event loop so cross-thread publishes can schedule."""
    global _loop
    _loop = loop


def set_run_user(user_id: str | None) -> contextvars.Token[str | None]:
    """Tag every subsequent publish() in this context with `user_id`.

    Returns the Token so the caller can `_user_var.reset(token)` to scope
    the override. Usually unnecessary — letting the var fall out of scope
    when the worker thread exits is enough.
    """
    return _user_var.set(user_id)


def subscribe(user_id: str = "") -> asyncio.Queue[Event]:
    """Subscribe to the stream. `user_id` filters incoming events to those
    tagged with the same user (plus untagged broadcasts). Pass "" to opt
    out of filtering (legacy / admin / CLI)."""
    q: asyncio.Queue[Event] = asyncio.Queue(maxsize=_QUEUE_MAX)
    with _subscribers_lock:
        _subscribers.append((q, user_id))
    return q


def unsubscribe(q: asyncio.Queue[Event]) -> None:
    with _subscribers_lock:
        for i, (sub_q, _uid) in enumerate(_subscribers):
            if sub_q is q:
                _subscribers.pop(i)
                return


def publish(event_type: str, data: dict[str, Any]) -> None:
    """Emit an event. Safe from any thread. No-op when no loop is attached
    (e.g. running via CLI script without the FastAPI stream up).

    The current `_user_var` value is stamped onto the event as
    `user_id`. Subscribers filter on it.
    """
    loop = _loop
    if loop is None:
        return
    user_id = _user_var.get()
    event: Event = {"type": event_type, "ts": time.time(), "data": data}
    if user_id:
        event["user_id"] = user_id
    with _subscribers_lock:
        subs = list(_subscribers)
    for q, sub_user in subs:
        # Subscriber sees the event if (a) it didn't filter, or (b) the
        # event has no user_id (broadcast), or (c) user_ids match.
        if sub_user and user_id and sub_user != user_id:
            continue
        try:
            asyncio.run_coroutine_threadsafe(_put_or_drop(q, event), loop)
        except RuntimeError:
            # Loop is stopping — drop silently.
            pass


async def _put_or_drop(q: asyncio.Queue[Event], event: Event) -> None:
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass

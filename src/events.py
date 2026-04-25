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

Last-Event-ID replay
--------------------
Each event gets a monotonic `event_id` from a global counter. Per-user
ring buffers (`_replay_buffers[user_id]`, max 200 entries) hold recent
events; `replay_since(user_id, last_id)` returns the slice newer than
`last_id`. The /stream endpoint reads the EventSource-managed
`Last-Event-ID` header on reconnect and replays before entering the
live loop, so a brief network blip is invisible to the frontend.
"""
from __future__ import annotations

import asyncio
import contextvars
import itertools
import threading
import time
from collections import deque
from typing import Any

Event = dict[str, Any]

# Subscriber tuple = (queue, user_id). user_id "" means "no scoping",
# receives every event. The stream endpoint always passes a real user_id;
# the empty-string case is reserved for tests and CLI consumers.
_subscribers: list[tuple[asyncio.Queue[Event], str]] = []
_subscribers_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None

_QUEUE_MAX = 5000

# Per-user ring buffer keyed by user_id; deque.maxlen bounds memory.
# 200 events covers ~30 minutes of pipeline activity at the busiest
# (persona generation fires ~1 event/persona, validation adds a few).
_REPLAY_MAX = 200
_replay_buffers: dict[str, deque[Event]] = {}

# Monotonic event-id counter. Wrap-around isn't a concern at hackathon
# scale (would take ~10^18 events). EventSource compares Last-Event-ID
# as a string but ints render the same; keeping the type as int for
# arithmetic correctness.
_event_id_counter = itertools.count(1)

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
    `user_id`. Subscribers filter on it. Each event gets a monotonic
    `event_id` and lands in the per-user replay buffer for SSE
    reconnect support.
    """
    loop = _loop
    if loop is None:
        return
    user_id = _user_var.get()
    event: Event = {
        "event_id": next(_event_id_counter),
        "type": event_type,
        "ts": time.time(),
        "data": data,
    }
    if user_id:
        event["user_id"] = user_id
        # Append to the owning user's replay buffer. Untagged broadcasts
        # aren't replayable — they're for system / admin signals and
        # don't have a target user to map them to.
        with _subscribers_lock:
            buf = _replay_buffers.setdefault(user_id, deque(maxlen=_REPLAY_MAX))
            buf.append(event)
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


def replay_since(user_id: str, last_event_id: int | None) -> list[Event]:
    """Return events from this user's buffer with event_id > last_event_id.

    `last_event_id is None` (fresh connect) returns []. A LEI older than
    the buffer's oldest entry returns the full buffer — the client lost
    a window of events to ring-buffer eviction; replaying everything we
    still have is the best the bus can do, and the frontend reconciler
    can tolerate it.
    """
    if not user_id or last_event_id is None:
        return []
    with _subscribers_lock:
        buf = _replay_buffers.get(user_id)
        if not buf:
            return []
        # deque iteration is O(n); for n=200 this is fine.
        return [e for e in buf if e["event_id"] > last_event_id]


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

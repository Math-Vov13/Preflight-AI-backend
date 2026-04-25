"""Thread-safe pub/sub event bus for runtime observability.

Runner threads call `publish(...)` — fire-and-forget, safe from any thread.
The SSE endpoint calls `subscribe()` from the async event loop and drains
per-subscriber queues. Events never block the runner; queue overflow drops
the oldest event.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

Event = dict[str, Any]

_subscribers: list[asyncio.Queue[Event]] = []
_subscribers_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None

_QUEUE_MAX = 5000


def attach_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Register the FastAPI event loop so cross-thread publishes can schedule."""
    global _loop
    _loop = loop


def subscribe() -> asyncio.Queue[Event]:
    q: asyncio.Queue[Event] = asyncio.Queue(maxsize=_QUEUE_MAX)
    with _subscribers_lock:
        _subscribers.append(q)
    return q


def unsubscribe(q: asyncio.Queue[Event]) -> None:
    with _subscribers_lock:
        if q in _subscribers:
            _subscribers.remove(q)


def publish(event_type: str, data: dict[str, Any]) -> None:
    """Emit an event. Safe from any thread. No-op when no loop is attached
    (e.g. running via CLI script without the FastAPI stream up)."""
    loop = _loop
    if loop is None:
        return
    event: Event = {"type": event_type, "ts": time.time(), "data": data}
    with _subscribers_lock:
        subs = list(_subscribers)
    for q in subs:
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

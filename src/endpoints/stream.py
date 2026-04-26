"""SSE endpoint that fans out events from the runtime event bus.

Auth: EventSource can't set custom headers, so we accept the JWT as a
`?token=<jwt>` query param and route it through `auth.resolve_user`. In
dev-local mode (no SUPABASE_JWT_SECRET) the token is ignored and every
subscriber lands on `dev_user_id`.

Isolation: the resolved user_id is passed to `events.subscribe(user_id)`
so this connection only receives events tagged with the same user (plus
untagged broadcasts). The pipeline tags every publish via the
`set_run_user` ContextVar — see events.py.

Replay: every published event has a monotonic `event_id` and lives in a
per-user ring buffer. We emit `id:` lines so EventSource tracks
Last-Event-ID automatically; on reconnect, the browser sends it back and
we replay the missed events before entering the live loop. Bridges the
network-blip gap with no frontend changes.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Header, Query
from fastapi.responses import StreamingResponse

from auth import resolve_user
from events import Event, replay_since, subscribe, unsubscribe

router = APIRouter(tags=["stream"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    # Disables buffering on reverse proxies (nginx, cloudflare, Next.js proxy).
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def _format(event: Event) -> str:
    eid = event.get("event_id")
    payload = json.dumps(event, ensure_ascii=False)
    if eid is not None:
        return f"id: {eid}\ndata: {payload}\n\n"
    return f"data: {payload}\n\n"


def _parse_last_event_id(raw: str | None) -> int | None:
    if not raw:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


@router.get("/stream")
async def stream(
    token: str | None = Query(default=None),
    last_event_id_header: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    user_id = resolve_user(token)
    last_event_id = _parse_last_event_id(last_event_id_header)
    queue = subscribe(user_id)

    async def gen():
        try:
            yield "event: connected\ndata: {}\n\n"
            # Replay first so the client sees the missed events in
            # publish-order before the live loop kicks in.
            for missed in replay_since(user_id, last_event_id):
                yield _format(missed)
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield _format(event)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            unsubscribe(queue)

    return StreamingResponse(gen(), media_type="text/event-stream", headers=_SSE_HEADERS)

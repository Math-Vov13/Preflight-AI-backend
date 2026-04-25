"""SSE endpoint that fans out events from the runtime event bus.

Auth: EventSource can't set custom headers, so we accept the JWT as a
`?token=<jwt>` query param and route it through `auth.resolve_user`. In
dev-local mode (no SUPABASE_JWT_SECRET) the token is ignored and every
subscriber lands on `dev_user_id`.

Isolation: the resolved user_id is passed to `events.subscribe(user_id)`
so this connection only receives events tagged with the same user (plus
untagged broadcasts). The pipeline tags every publish via the
`set_run_user` ContextVar — see events.py.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from auth import resolve_user
from events import subscribe, unsubscribe

router = APIRouter(tags=["stream"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    # Disables buffering on reverse proxies (nginx, cloudflare, Next.js proxy).
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


@router.get("/stream")
async def stream(token: str | None = Query(default=None)) -> StreamingResponse:
    user_id = resolve_user(token)
    queue = subscribe(user_id)

    async def gen():
        try:
            yield "event: connected\ndata: {}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            unsubscribe(queue)

    return StreamingResponse(gen(), media_type="text/event-stream", headers=_SSE_HEADERS)

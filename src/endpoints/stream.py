"""SSE endpoint that fans out events from the runtime event bus."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from events import subscribe, unsubscribe

router = APIRouter(tags=["stream"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    # Disables buffering on reverse proxies (nginx, cloudflare, Next.js proxy).
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


@router.get("/stream")
async def stream() -> StreamingResponse:
    queue = subscribe()

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

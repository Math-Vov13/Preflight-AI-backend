"""POST /stt/transcribe — Gradium speech-to-text proxy.

The Next.js frontend records audio in the browser, resamples it to PCM 24
kHz / 16-bit / mono, and POSTs the raw bytes here as a multipart `audio`
field. We feed those bytes to the official `gradium` Python SDK over its
streaming STT API and return the joined transcript.
"""
from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stt", tags=["stt"])

MAX_AUDIO_BYTES = 25 * 1024 * 1024
# 1920 samples * 2 bytes (Int16) = 3840 bytes — Gradium's recommended
# 80 ms chunk at 24 kHz / 16-bit / mono.
PCM_CHUNK_BYTES = 3840

_client: Any | None = None
_client_init_error: str | None = None


def _get_client() -> Any:
    global _client, _client_init_error
    if _client is not None:
        return _client
    api_key = os.getenv("GRADIUM_API_KEY")
    if not api_key:
        _client_init_error = "GRADIUM_API_KEY not set"
        raise HTTPException(status_code=503, detail=_client_init_error)
    try:
        import gradium  # type: ignore
        _client = gradium.client.GradiumClient(api_key=api_key)
        return _client
    except ImportError as e:
        _client_init_error = f"gradium SDK not installed: {e}"
        raise HTTPException(status_code=503, detail=_client_init_error) from e
    except Exception as e:
        _client_init_error = f"Gradium client init failed: {e}"
        raise HTTPException(status_code=503, detail=_client_init_error) from e


async def _audio_chunks(buf: bytes, chunk_size: int = PCM_CHUNK_BYTES) -> AsyncIterator[bytes]:
    for i in range(0, len(buf), chunk_size):
        yield buf[i : i + chunk_size]


def _extract_text(message: Any) -> str:
    """Best-effort text extraction from a Gradium STT message.

    The SDK's `iter_text` may yield strings, dicts, or pydantic-ish objects
    depending on the version. Probe in that order.
    """
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        for key in ("text", "transcript", "content"):
            value = message.get(key)
            if isinstance(value, str) and value:
                return value
        return ""
    for attr in ("text", "transcript", "content"):
        value = getattr(message, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> JSONResponse:
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=422, detail="Empty audio")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio exceeds 25 MB")

    client = _get_client()

    try:
        stream = await client.stt_stream(
            {"model_name": "default", "input_format": "pcm"},
            _audio_chunks(data),
        )
    except Exception as e:
        logger.exception("Gradium stt_stream init failed")
        return JSONResponse(status_code=502, content={"error": f"Gradium init failed: {e}"})

    parts: list[str] = []
    try:
        async for message in stream.iter_text():
            text = _extract_text(message)
            if text:
                parts.append(text)
    except Exception as e:
        logger.exception("Gradium stt_stream iteration failed")
        return JSONResponse(status_code=502, content={"error": f"Gradium stream failed: {e}"})

    transcript = " ".join(p.strip() for p in parts if p.strip())
    return JSONResponse(content={"transcript": transcript})

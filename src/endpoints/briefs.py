"""POST /api/briefs/parse — accept a PDF / md / txt upload, return plain text.

Used by the /live page to let users drop a cahier-des-charges PDF (or any
supporting doc) and have its content appended to the product Brief before
firing a run.
"""
from __future__ import annotations

import logging
from typing import Annotated

import fitz  # PyMuPDF
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from auth import CurrentUser
from rate_limit import TokenBucket

router = APIRouter(tags=["briefs"])
logger = logging.getLogger(__name__)

MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
MAX_CHARS = 60_000  # truncate pathological inputs so the LLM stays in context

# 10 parses/min per user — light protection against PDF-parse abuse.
# PyMuPDF on a 100-page PDF takes ~1 s, so a tight loop could pin a CPU
# easily; a token bucket with burst=10 and refill=10/min lets the
# common case (drop 3 docs into the composer at once) through without
# friction.
_PARSE_BUCKET = TokenBucket(capacity=10, refill_per_sec=10 / 60)


class ParsedDocument(BaseModel):
    filename: str
    mime_type: str
    content: str
    n_pages: int | None = None
    n_chars: int
    truncated: bool


def _sniff_type(filename: str, declared_mime: str) -> str:
    name = filename.lower()
    if declared_mime == "application/pdf" or name.endswith(".pdf"):
        return "pdf"
    if (
        declared_mime.startswith("text/")
        or name.endswith(".md")
        or name.endswith(".markdown")
        or name.endswith(".txt")
    ):
        return "text"
    return "unknown"


def _decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            from charset_normalizer import from_bytes

            best = from_bytes(data).best()
            if best is not None:
                return str(best)
        except Exception as e:
            logger.warning("charset-normalizer fell through: %s", e)
    return data.decode("utf-8", errors="replace")


@router.post("/briefs/parse", response_model=ParsedDocument)
async def parse_document(
    user: CurrentUser,
    file: Annotated[UploadFile, File(description="PDF, markdown, or plain text")],
) -> ParsedDocument:
    retry_after = _PARSE_BUCKET.try_consume(user)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="too many parse requests; slow down",
            headers={"Retry-After": str(retry_after)},
        )
    # `user` was used for the rate-limit key; the parse itself is
    # stateless (no per-user storage of uploads). Auth dependency
    # presence still ensures unauthenticated requests can't even reach
    # the bucket check.
    data = await file.read()
    size = len(data)
    if size == 0:
        raise HTTPException(status_code=400, detail="empty upload")
    if size > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file too large ({size // 1024 // 1024} MB > {MAX_BYTES // 1024 // 1024} MB)",
        )

    name = file.filename or "document"
    mime = file.content_type or ""
    kind = _sniff_type(name, mime)

    content = ""
    n_pages: int | None = None

    if kind == "pdf":
        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid PDF: {e}") from e
        try:
            n_pages = doc.page_count
            pages: list[str] = []
            for i in range(n_pages):
                pages.append(doc[i].get_text("text"))
            content = "\n\n".join(pages)
        finally:
            doc.close()
    elif kind == "text":
        content = _decode_text(data)
    else:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported type: {mime or name} (PDF / md / txt only)",
        )

    content = content.strip()
    n_chars_raw = len(content)
    truncated = False
    if n_chars_raw > MAX_CHARS:
        content = content[:MAX_CHARS]
        truncated = True

    logger.info(
        "parsed %s (%s) — %d chars%s%s",
        name,
        kind,
        n_chars_raw,
        f" / {n_pages} pages" if n_pages is not None else "",
        " [truncated]" if truncated else "",
    )

    return ParsedDocument(
        filename=name,
        mime_type=mime or ("application/pdf" if kind == "pdf" else "text/plain"),
        content=content,
        n_pages=n_pages,
        n_chars=n_chars_raw,
        truncated=truncated,
    )

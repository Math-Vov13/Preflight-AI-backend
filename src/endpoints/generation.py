import asyncio
import base64
import binascii
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.base.llms.types import ImageBlock, TextBlock
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    RateLimitError,
)
from pydantic import BaseModel

from models.cache_redis.chat_history import load_history, save_history
from models.cache_redis.client import async_client as redis_async_client
from models.supabase_storage.upload_files import upload_file_to_supabase
from rag.server import (
    LLMTextDelta,
    LLMToolCallsAnnounced,
    LLMTurnEnd,
    LLMTurnStart,
    ToolResult,
    build_workflow,
)
from schema.generation_streaming import (
    ChunkEnd,
    ChunkMessage,
    ChunkStart,
    ChunkToolEnd,
    ContentModeration,
    ErrorResponse,
)


def _classify_error(exc: Exception) -> ErrorResponse:
    """Map exceptions to a stable, client-friendly ErrorResponse."""
    if isinstance(exc, AuthenticationError):
        return ErrorResponse(
            error_type="auth_error",
            error="The upstream LLM rejected the API key. Check SILICONFLOW_API_KEY and SILICONFLOW_BASE_URL.",
            code=401,
        )
    if isinstance(exc, PermissionDeniedError):
        return ErrorResponse(
            error_type="auth_error",
            error="The API key does not have permission for this model or endpoint.",
            code=403,
        )
    if isinstance(exc, RateLimitError):
        return ErrorResponse(
            error_type="rate_limit",
            error="Upstream rate limit reached. Retry in a moment.",
            code=429,
        )
    if isinstance(exc, BadRequestError):
        return ErrorResponse(
            error_type="bad_request",
            error=f"Upstream rejected the request: {exc}",
            code=400,
        )
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return ErrorResponse(
            error_type="network_error",
            error="Could not reach the upstream LLM. Check network and SILICONFLOW_BASE_URL.",
            code=None,
        )
    if isinstance(exc, APIStatusError):
        return ErrorResponse(
            error_type="upstream_error",
            error=f"Upstream LLM returned an error: {exc}",
            code=getattr(exc, "status_code", None),
        )
    return ErrorResponse(
        error_type="internal_error",
        error=str(exc) or exc.__class__.__name__,
        code=None,
    )

router = APIRouter()


class FileItem(BaseModel):
    name: str
    size: int
    mimeType: str
    type: str
    base64: str


class GenerationRequest(BaseModel):
    prompt: str
    history_id: Optional[str] = None
    files: Optional[list[FileItem]] = None


def _build_user_message(prompt: str, files: Optional[list[FileItem]]) -> ChatMessage:
    if not files:
        return ChatMessage(role=MessageRole.USER, content=prompt)
    blocks: list = [TextBlock(text=prompt)]
    for file in files:
        blocks.append(ImageBlock(url=file.base64))
    return ChatMessage(role=MessageRole.USER, blocks=blocks)


def _decode_base64_payload(payload: str) -> bytes | None:
    if not payload:
        return None
    raw = payload.split(",", 1)[1] if payload.startswith("data:") else payload
    try:
        return base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError):
        return None


async def _persist_attachments(files: Optional[list[FileItem]]) -> list[dict]:
    if not files:
        return []

    def _upload_one(item: FileItem) -> dict | None:
        content = _decode_base64_payload(item.base64)
        if content is None:
            print(f"Skipping {item.name}: invalid base64 payload.", flush=True)
            return None
        key = upload_file_to_supabase(
            file_content=content,
            file_name=item.name,
            content_type=item.mimeType or "application/octet-stream",
            role="upload",
        )
        if key is None:
            return None
        return {
            "name": item.name,
            "mime_type": item.mimeType,
            "size": item.size,
            "storage_key": key,
        }

    results = await asyncio.gather(
        *(asyncio.to_thread(_upload_one, f) for f in files)
    )
    return [r for r in results if r is not None]


def _sse(payload: BaseModel, event_name: str | None = None) -> str:
    prefix = f"event: {event_name}\n" if event_name else ""
    return f"{prefix}data: {payload.model_dump_json()}\n\n"


@router.post("/")
async def create_generation_json(request: GenerationRequest) -> StreamingResponse:
    request_id = "req-" + str(uuid4())
    session_id = request.history_id or request_id

    history = await load_history(redis_async_client, session_id)
    attachments = await _persist_attachments(request.files)
    persisted_user_message = ChatMessage(
        role=MessageRole.USER,
        content=request.prompt,
        additional_kwargs={"attachments": attachments} if attachments else {},
    )
    llm_user_message = _build_user_message(request.prompt, request.files)

    workflow = build_workflow(collection_name="1234")  # TODO: per-user collection
    handler = workflow.run(messages=history + [llm_user_message])

    async def event_stream():
        yield _sse(ContentModeration(request_id=request_id, moderate=None))

        try:
            async for evt in handler.stream_events():
                if isinstance(evt, LLMTurnStart):
                    yield _sse(
                        ChunkStart(
                            run_id=evt.run_id,
                            graph_node={
                                "step": 0,
                                "node": "generation_task",
                                "_provider": "siliconflow",
                                "_name": evt.model,
                                "_type": "chat",
                            },
                            params={
                                "temperature": 0.3,
                                "max_tokens": 7000,
                                "top_p": 0,
                                "presence_penalty": 0,
                                "frequency_penalty": 0,
                            },
                        ),
                        event_name="delta",
                    )
                elif isinstance(evt, LLMTextDelta):
                    yield _sse(
                        ChunkMessage(
                            run_id=evt.run_id,
                            parts=[{"type": "text", "text": evt.text}],
                        ),
                        event_name="delta",
                    )
                elif isinstance(evt, LLMToolCallsAnnounced):
                    yield _sse(
                        ChunkMessage(
                            run_id=evt.run_id,
                            parts=[{"type": "text", "text": ""}],
                            tool_calls=evt.tool_calls,
                        ),
                        event_name="delta",
                    )
                elif isinstance(evt, LLMTurnEnd):
                    yield _sse(
                        ChunkEnd(
                            run_id=evt.run_id,
                            response_metadata=evt.response_metadata,
                        ),
                        event_name="delta",
                    )
                elif isinstance(evt, ToolResult):
                    yield _sse(
                        ChunkToolEnd(
                            run_id=evt.run_id,
                            tool_id=evt.tool_id,
                            tool_name=evt.tool_name,
                            data={"output": evt.output, "input": evt.input},
                        )
                    )

            final_message: ChatMessage = await handler
            history.extend([persisted_user_message, final_message])
            await save_history(redis_async_client, session_id, history)

        except Exception as exc:
            err = _classify_error(exc)
            print(
                f"Error during generation: type={err.error_type} code={err.code} exc={type(exc).__name__}: {exc}",
                flush=True,
            )
            yield _sse(err)

        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

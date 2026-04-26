import asyncio
import base64
import binascii
import logging
from typing import Any, Optional
from uuid import uuid4

import fitz  # PyMuPDF — used to extract text from PDF attachments
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
from models.siliconflow import (
    StreamEnd,
    StreamTextDelta,
    StreamToolCall,
    client as siliconflow_client,
)
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
from sim_config import settings


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
logger = logging.getLogger(__name__)

# Documents are inlined into the prompt; cap each one so a 500-page PDF
# can't blow past the LLM context window all by itself.
_DOC_INLINE_MAX_CHARS = 60_000


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
    # ---- Orchestrator / structured-output extension ------------------------
    # Custom system prompt overriding the workflow's default. Required for
    # the orchestrator path; ignored unless the model needs context the chat
    # workflow doesn't already inject.
    system: Optional[str] = None
    # OpenAI/Anthropic-style tool definitions to forward to the LLM. When
    # set together with `tools_passthrough=True`, the endpoint bypasses the
    # LlamaIndex workflow entirely and streams a single LLM turn.
    tools: Optional[list[dict[str, Any]]] = None
    # Either "auto", "none", "required", or a typed object like
    # {"type": "tool", "name": "<name>"}. When the type is "tool" or
    # "function" we coerce to the OpenAI shape automatically — the FE
    # passes Anthropic-style and the backend rewrites.
    tool_choice: Optional[dict[str, Any] | str] = None
    # When True, the endpoint runs in "structured output" mode: tools are
    # advertised to the LLM but never executed downstream. The stream ends
    # as soon as the LLM emits its tool call(s). Used by the preflight
    # orchestrator to extract a JSON payload from the chat LLM without
    # triggering a tool-execution loop.
    tools_passthrough: bool = False


def _decode_base64_payload(payload: str) -> bytes | None:
    if not payload:
        return None
    raw = payload.split(",", 1)[1] if payload.startswith("data:") else payload
    try:
        return base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError):
        return None


def _is_image(item: FileItem) -> bool:
    if item.mimeType and item.mimeType.startswith("image/"):
        return True
    if item.base64.startswith("data:image/"):
        return True
    return False


def _is_pdf(item: FileItem) -> bool:
    if item.mimeType == "application/pdf":
        return True
    return item.name.lower().endswith(".pdf")


def _is_text_doc(item: FileItem) -> bool:
    if item.mimeType and (
        item.mimeType.startswith("text/") or item.mimeType == "application/json"
    ):
        return True
    name = item.name.lower()
    return name.endswith((".txt", ".md", ".markdown"))


def _decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _extract_pdf_text(data: bytes) -> str:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        return "\n\n".join(doc[i].get_text("text") for i in range(doc.page_count))
    finally:
        doc.close()


def _doc_block(item: FileItem, text: str) -> TextBlock:
    body = text.strip()
    truncated = ""
    if len(body) > _DOC_INLINE_MAX_CHARS:
        body = body[:_DOC_INLINE_MAX_CHARS]
        truncated = "\n\n[...document truncated]"
    return TextBlock(
        text=f"\n\n--- attached document: {item.name} ---\n{body}{truncated}\n--- end {item.name} ---",
    )


def _build_user_message(prompt: str, files: Optional[list[FileItem]]) -> ChatMessage:
    if not files:
        return ChatMessage(role=MessageRole.USER, content=prompt)

    blocks: list = [TextBlock(text=prompt)]
    for file in files:
        # Images are passed through to the multimodal model as-is — the URL
        # is the data URL the frontend already produced via FileReader.
        if _is_image(file):
            blocks.append(ImageBlock(url=file.base64))
            continue

        # Documents (PDF / MD / TXT) get parsed to text and inlined as a
        # TextBlock. SiliconFlow rejects non-image bytes inside ImageBlock
        # with "图片输入格式/解析错误" (code 20015), which is what the user
        # was hitting before this branch existed.
        payload = _decode_base64_payload(file.base64)
        if payload is None:
            logger.warning("attachment %s: invalid base64; skipping", file.name)
            continue

        try:
            if _is_pdf(file):
                content = _extract_pdf_text(payload)
            elif _is_text_doc(file):
                content = _decode_text(payload)
            else:
                logger.warning(
                    "attachment %s (%s): unsupported type for chat; skipping",
                    file.name,
                    file.mimeType,
                )
                continue
        except Exception as exc:  # noqa: BLE001 — surface as a soft skip, not a 500
            logger.warning("attachment %s: parse failed (%s); skipping", file.name, exc)
            continue

        if content.strip():
            blocks.append(_doc_block(file, content))

    return ChatMessage(role=MessageRole.USER, blocks=blocks)


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


def _normalize_tool_choice(
    tool_choice: dict[str, Any] | str | None,
) -> dict[str, Any] | str | None:
    """Coerce Anthropic-style tool_choice to OpenAI shape.

    The FE sends `{type: "tool", name: "<x>"}` (Anthropic). OpenAI expects
    `{type: "function", function: {name: "<x>"}}`. Pass-through for plain
    strings ("auto", "none", "required") and for already-OpenAI shapes.
    """
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    t = tool_choice.get("type")
    if t == "tool" and "name" in tool_choice:
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return tool_choice


def _normalize_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce Anthropic-style tool defs ({name, description, input_schema})
    to OpenAI ({type: "function", function: {name, description, parameters}}).
    """
    normalized: list[dict[str, Any]] = []
    for t in tools:
        if t.get("type") == "function" and "function" in t:
            normalized.append(t)
            continue
        # Anthropic-style → OpenAI
        if "name" in t and "input_schema" in t:
            normalized.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t["input_schema"],
                },
            })
            continue
        # Already OpenAI without the wrapper, or unknown — pass through.
        normalized.append(t)
    return normalized


async def _stream_tools_passthrough(
    request: GenerationRequest,
    request_id: str,
) -> StreamingResponse:
    """Single-turn LLM call that surfaces text + tool calls to the client.

    Used by the preflight orchestrator. The LLM is given a `system` prompt,
    the user's `prompt`, a `tools` list, and a forced `tool_choice`. We
    stream every text delta as a `chat_model_stream` event with `parts`,
    and emit one terminal `chat_model_stream` carrying the assembled
    `tool_calls` once the model finishes. No tool is executed.
    """
    if not request.tools:
        err = ErrorResponse(
            error_type="bad_request",
            error="tools_passthrough requires a non-empty tools list.",
            code=400,
        )
        async def _err_stream():
            yield _sse(err)
            yield "data: [DONE]\n\n"
        return StreamingResponse(_err_stream(), media_type="text/event-stream")

    messages: list[dict[str, Any]] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.append({"role": "user", "content": request.prompt})

    tools = _normalize_tools(request.tools)
    tool_choice = _normalize_tool_choice(request.tool_choice)
    # Use the orchestrator-specific model (NOT chat_model). DeepSeek-V3's
    # tool calling is broken on SiliconFlow's adapter — its native
    # `<｜tool▁call▁begin｜>` tokens leak into `function.arguments` raw,
    # producing empty parsed args. Qwen2.5-72B-Instruct is the safe default.
    model_id = settings().orchestrator_model
    run_id = "gen-" + str(uuid4())

    async def event_stream():
        yield _sse(ContentModeration(request_id=request_id, moderate=None))
        yield _sse(
            ChunkStart(
                run_id=run_id,
                graph_node={
                    "step": 0,
                    "node": "orchestrator_passthrough",
                    "_provider": "siliconflow",
                    "_name": model_id,
                    "_type": "chat",
                },
                params={"temperature": 0.3, "tool_choice": tool_choice},
            ),
            event_name="delta",
        )

        try:
            # The streaming SiliconFlow generator is sync; iterate it on a
            # worker thread so the asyncio event loop stays free for the
            # SSE send pump.
            loop = asyncio.get_running_loop()

            def _drain() -> list:
                events = []
                for ev in siliconflow_client().chat_stream_with_tools(
                    model=model_id,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                ):
                    events.append(ev)
                return events

            # We collect first, then emit — the OpenAI SDK's sync iterator
            # doesn't release the GIL between chunks in a way that
            # interleaves cleanly with `yield` here. Latency cost: the
            # passthrough path is short (intro + tool call), so buffering
            # the whole turn before emitting is acceptable. If this ever
            # needs true streaming, switch to the async OpenAI client.
            events = await loop.run_in_executor(None, _drain)

            tool_calls_payload: list[dict[str, Any]] = []
            for ev in events:
                if isinstance(ev, StreamTextDelta):
                    if ev.text:
                        yield _sse(
                            ChunkMessage(
                                run_id=run_id,
                                parts=[{"type": "text", "text": ev.text}],
                            ),
                            event_name="delta",
                        )
                elif isinstance(ev, StreamToolCall):
                    tool_calls_payload.append({
                        "id": ev.id,
                        "name": ev.name,
                        "args": ev.arguments,
                        # Raw JSON string surfaced for FE fallback parsing.
                        # Some SiliconFlow models ignore the input_schema and
                        # return empty parsed args even though the raw text
                        # contains valid JSON; the FE re-parses from this.
                        "raw_arguments": ev.raw_arguments,
                    })
                elif isinstance(ev, StreamEnd):
                    if tool_calls_payload:
                        yield _sse(
                            ChunkMessage(
                                run_id=run_id,
                                parts=[{"type": "text", "text": ""}],
                                tool_calls=tool_calls_payload,
                            ),
                            event_name="delta",
                        )
                    yield _sse(
                        ChunkEnd(
                            run_id=run_id,
                            response_metadata={
                                "finish_reason": ev.finish_reason,
                                "n_tool_calls": len(tool_calls_payload),
                            },
                        ),
                        event_name="delta",
                    )

        except Exception as exc:
            err = _classify_error(exc)
            print(
                f"Error during passthrough generation: type={err.error_type} code={err.code} exc={type(exc).__name__}: {exc}",
                flush=True,
            )
            yield _sse(err)
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/")
async def create_generation_json(request: GenerationRequest) -> StreamingResponse:
    request_id = "req-" + str(uuid4())
    session_id = request.history_id or request_id

    # Orchestrator / structured-output path — no chat history, no RAG, no
    # tool execution loop. Just one LLM turn that surfaces text + tool calls.
    if request.tools_passthrough:
        return await _stream_tools_passthrough(request, request_id)

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

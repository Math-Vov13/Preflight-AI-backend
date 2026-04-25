"""LlamaIndex Workflow that drives the chat: RAG retrieve → stream LLM → tool loop."""

from typing import Any
from uuid import uuid4

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from pydantic import BaseModel

from models.vc_qdrant.utils import collection_exists, openai_ef, query_documents
from rag.config import llm
from rag.tools import ALL_TOOLS

SYSTEM_PROMPT = open("src/docs/GEMINI_SYSTEM_PROMPT.md").read()
USER_DOCS_TEMPLATE = open("src/docs/USER_MESSAGE_WITH_DOCS_CONTEXT.md").read()


# ---- streaming events (consumed by the FastAPI endpoint) ----

class LLMTurnStart(Event):
    run_id: str
    model: str


class LLMTextDelta(Event):
    run_id: str
    text: str


class LLMToolCallsAnnounced(Event):
    run_id: str
    tool_calls: list[dict[str, Any]]


class LLMTurnEnd(Event):
    run_id: str
    response_metadata: dict[str, Any] = {}


class ToolResult(Event):
    run_id: str
    tool_id: str
    tool_name: str
    output: str
    input: dict[str, Any]


# ---- internal flow events ----

class GenerateLLMEvent(Event):
    pass


class ToolCallsEvent(Event):
    tool_calls: list[Any]


def _jsonable(obj: Any) -> Any:
    # openai's BaseModel sets defer_build=True, so its serializer stays as
    # MockValSer until forced. Nesting these objects inside another pydantic
    # model's dict[str, Any] field causes model_dump_json to raise
    # PydanticSerializationError. Convert them to plain dicts up front.
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, BaseModel):
        to_dict = getattr(obj, "to_dict", None)
        if callable(to_dict):
            try:
                return to_dict(mode="json")
            except TypeError:
                try:
                    return to_dict()
                except Exception:
                    pass
        try:
            return obj.model_dump(mode="json")
        except Exception:
            return str(obj)
    return obj


def _message_text(msg: ChatMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        parts = []
        for block in msg.content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts).strip()
    return ""


class ChatWorkflow(Workflow):
    """Single-turn workflow: optional RAG retrieval, streamed LLM, tool loop."""

    def __init__(self, *, collection_name: str | None = None, timeout: int = 600, **kwargs):
        super().__init__(timeout=timeout, **kwargs)
        self._llm = llm
        self._tools = ALL_TOOLS
        self._tools_by_name = {t.metadata.name: t for t in ALL_TOOLS}
        self._collection_name = collection_name

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> GenerateLLMEvent:
        messages: list[ChatMessage] = list(ev.messages)
        self._maybe_augment_with_rag(messages)
        full = [ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT)] + messages
        await ctx.store.set("messages", full)
        return GenerateLLMEvent()

    def _maybe_augment_with_rag(self, messages: list[ChatMessage]) -> None:
        if not self._collection_name or not messages:
            return
        last = messages[-1]
        if last.role != MessageRole.USER:
            return
        query = _message_text(last)
        if not query:
            return
        try:
            if not collection_exists(self._collection_name):
                return
            embeddings = openai_ef(input=query)
            if not embeddings:
                return
            docs = query_documents(self._collection_name, embeddings[0], limit=5)
            docs = [d for d in docs if d]
            if not docs:
                return
            context = "\n\n".join(docs).strip()
            messages[-1] = ChatMessage(
                role=MessageRole.USER,
                content=USER_DOCS_TEMPLATE.format(DOCUMENTS_CONTEXT=context, USER_PROMPT=query),
            )
        except Exception as exc:
            print(f"RAG retrieval skipped: {exc}", flush=True)

    @step
    async def generate(self, ctx: Context, ev: GenerateLLMEvent) -> ToolCallsEvent | StopEvent:
        messages: list[ChatMessage] = await ctx.store.get("messages")
        run_id = str(uuid4())
        ctx.write_event_to_stream(
            LLMTurnStart(run_id=run_id, model=self._llm.metadata.model_name)
        )

        stream = await self._llm.astream_chat_with_tools(self._tools, chat_history=messages)
        last_response = None
        async for partial in stream:
            last_response = partial
            delta = partial.delta or ""
            if delta:
                ctx.write_event_to_stream(LLMTextDelta(run_id=run_id, text=delta))

        if last_response is None:
            ctx.write_event_to_stream(LLMTurnEnd(run_id=run_id))
            return StopEvent(result=ChatMessage(role=MessageRole.ASSISTANT, content=""))

        assistant_msg = last_response.message
        messages.append(assistant_msg)
        await ctx.store.set("messages", messages)

        tool_calls = self._llm.get_tool_calls_from_response(
            last_response, error_on_no_tool_call=False
        )
        if tool_calls:
            ctx.write_event_to_stream(
                LLMToolCallsAnnounced(
                    run_id=run_id,
                    tool_calls=[
                        {
                            "id": tc.tool_id,
                            "name": tc.tool_name,
                            "args": _jsonable(tc.tool_kwargs),
                        }
                        for tc in tool_calls
                    ],
                )
            )

        ctx.write_event_to_stream(
            LLMTurnEnd(
                run_id=run_id,
                response_metadata=_jsonable(
                    getattr(assistant_msg, "additional_kwargs", {}) or {}
                ),
            )
        )

        if not tool_calls:
            return StopEvent(result=assistant_msg)
        return ToolCallsEvent(tool_calls=tool_calls)

    @step
    async def call_tools(self, ctx: Context, ev: ToolCallsEvent) -> GenerateLLMEvent:
        messages: list[ChatMessage] = await ctx.store.get("messages")
        for tc in ev.tool_calls:
            tool = self._tools_by_name.get(tc.tool_name)
            if tool is None:
                output = f"Tool '{tc.tool_name}' is not available."
            else:
                try:
                    result = await tool.acall(**tc.tool_kwargs)
                    output = str(result)
                except Exception as exc:
                    output = f"Tool error: {exc}"

            ctx.write_event_to_stream(
                ToolResult(
                    run_id=str(uuid4()),
                    tool_id=tc.tool_id,
                    tool_name=tc.tool_name,
                    output=output,
                    input=tc.tool_kwargs,
                )
            )

            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL,
                    content=output,
                    additional_kwargs={"tool_call_id": tc.tool_id, "name": tc.tool_name},
                )
            )

        await ctx.store.set("messages", messages)
        return GenerateLLMEvent()


def build_workflow(collection_name: str | None = None) -> ChatWorkflow:
    return ChatWorkflow(collection_name=collection_name)

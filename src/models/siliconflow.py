"""SiliconFlow LLM client (OpenAI-compatible endpoint)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

from openai import OpenAI

from sim_config import settings


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    raw: Any


@dataclass
class StreamTextDelta:
    """Incremental text chunk emitted during a streaming chat call."""
    kind: Literal["text"] = "text"
    text: str = ""


@dataclass
class StreamToolCall:
    """One complete tool call assembled from streaming deltas. Emitted when
    the model finishes producing the call's arguments JSON."""
    kind: Literal["tool_call"] = "tool_call"
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""


@dataclass
class StreamEnd:
    """Sentinel emitted once the stream finishes (after all text + tool calls)."""
    kind: Literal["end"] = "end"
    finish_reason: str | None = None


StreamEvent = StreamTextDelta | StreamToolCall | StreamEnd


class SiliconFlowClient:
    def __init__(self) -> None:
        s = settings()
        self._client = OpenAI(
            api_key=s.siliconflow_api_key, base_url=s.siliconflow_base_url
        )

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> ChatResult:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        usage = resp.usage
        return ChatResult(
            text=choice.message.content or "",
            model=resp.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            raw=resp,
        )

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(input=texts, model=model)
        return [d.embedding for d in resp.data]

    def chat_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ):
        """Yield content delta strings from the streaming API.

        The caller is responsible for framing (SSE, WebSocket, etc).
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        resp = self._client.chat.completions.create(**kwargs)
        for chunk in resp:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def chat_stream_with_tools(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any] | str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> Iterator[StreamEvent]:
        """Stream a chat completion with tool-calling support.

        Yields `StreamTextDelta` for each content delta, then one
        `StreamToolCall` per fully-assembled tool call (the OpenAI-compatible
        API streams tool-call args as JSON fragments — we buffer and parse
        them per-call), and finally a `StreamEnd` sentinel.

        The OpenAI Responses API for chat completions delivers tool calls
        as deltas indexed by position; we accumulate `arguments` per index
        and only emit the `StreamToolCall` once the parsed JSON is valid.
        Malformed JSON falls through with `arguments={}` and the raw text
        in `raw_arguments` so the caller can decide whether to surface or
        retry.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "tools": tools,
        }
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        # Per-tool-call accumulators keyed by the streaming `index` field.
        tool_buffers: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None

        resp = self._client.chat.completions.create(**kwargs)
        for chunk in resp:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if delta is not None:
                if delta.content:
                    yield StreamTextDelta(text=delta.content)

                tcalls = getattr(delta, "tool_calls", None) or []
                for tc in tcalls:
                    idx = getattr(tc, "index", 0) or 0
                    buf = tool_buffers.setdefault(
                        idx, {"id": "", "name": "", "arguments": ""},
                    )
                    if tc.id:
                        buf["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if fn.name:
                            buf["name"] = fn.name
                        if fn.arguments:
                            buf["arguments"] += fn.arguments

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        # Emit the assembled tool calls (in index order) before the end
        # sentinel so the caller can act on them deterministically.
        import logging as _logging
        _log = _logging.getLogger(__name__)
        for idx in sorted(tool_buffers.keys()):
            buf = tool_buffers[idx]
            raw_args = buf["arguments"]
            try:
                parsed = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError as e:
                _log.warning(
                    "tool_call %r returned malformed JSON args (%s) — raw=%r",
                    buf["name"], e, raw_args[:500],
                )
                parsed = {}
            # Surface a warning if args are empty or schema-incomplete — this
            # is the most common failure mode with smaller models that ignore
            # the function-calling input_schema.
            if not raw_args:
                _log.warning(
                    "tool_call %r returned EMPTY arguments — model likely "
                    "ignored the schema. Consider switching CHAT_MODEL to a "
                    "model with stronger tool_choice compliance.",
                    buf["name"],
                )
            else:
                _log.info(
                    "tool_call %r args ok (len=%d, keys=%s)",
                    buf["name"], len(raw_args),
                    list(parsed.keys()) if isinstance(parsed, dict) else "non-dict",
                )
            yield StreamToolCall(
                id=buf["id"],
                name=buf["name"],
                arguments=parsed if isinstance(parsed, dict) else {"_value": parsed},
                raw_arguments=raw_args,
            )

        yield StreamEnd(finish_reason=finish_reason)


_client: SiliconFlowClient | None = None


def client() -> SiliconFlowClient:
    global _client
    if _client is None:
        _client = SiliconFlowClient()
    return _client

"""SiliconFlow LLM client (OpenAI-compatible endpoint)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from sim_config import settings


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    raw: Any


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


_client: SiliconFlowClient | None = None


def client() -> SiliconFlowClient:
    global _client
    if _client is None:
        _client = SiliconFlowClient()
    return _client

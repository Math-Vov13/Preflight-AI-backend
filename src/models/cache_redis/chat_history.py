import json
from typing import Any

from llama_index.core.llms import ChatMessage, MessageRole
from redis.asyncio import Redis as AsyncRedis

_KEY = "chat_history:{session_id}"
_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days


def _extract_text(msg: ChatMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        return "".join(
            block.get("text", "")
            for block in msg.content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(msg.content) if msg.content else ""


def _serialize(msg: ChatMessage) -> dict[str, Any]:
    role = msg.role
    return {
        "role": role.value if isinstance(role, MessageRole) else str(role),
        "content": _extract_text(msg),
        "additional_kwargs": dict(msg.additional_kwargs or {}),
    }


def _deserialize(data: dict[str, Any]) -> ChatMessage:
    raw_role = data.get("role", "user")
    if isinstance(raw_role, str) and raw_role.startswith("MessageRole."):
        raw_role = raw_role.split(".", 1)[1].lower()
    try:
        role = MessageRole(raw_role)
    except ValueError:
        role = MessageRole.USER
    return ChatMessage(
        role=role,
        content=data.get("content", ""),
        additional_kwargs=data.get("additional_kwargs") or {},
    )


async def load_history(client: AsyncRedis | None, session_id: str) -> list[ChatMessage]:
    if client is None:
        return []
    raw = await client.get(_KEY.format(session_id=session_id))
    if not raw:
        return []
    return [_deserialize(d) for d in json.loads(raw)]


async def save_history(
    client: AsyncRedis | None,
    session_id: str,
    messages: list[ChatMessage],
) -> None:
    if client is None:
        return
    payload = json.dumps([_serialize(m) for m in messages])
    await client.set(_KEY.format(session_id=session_id), payload, ex=_TTL_SECONDS)

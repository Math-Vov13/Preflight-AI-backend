from pydantic import BaseModel, Field
from typing import Any
from enum import StrEnum
from datetime import datetime

class DeltaType(StrEnum):
    REQUEST_START = "request_start"
    REQUEST_END = "generation_end"

    REQUEST_MODERATED = "content_moderation"
    
    REQUEST_CHUNK_START = "chat_model_start"
    REQUEST_CHUNK = "chat_model_stream"
    REQUEST_CHUNK_END = "chat_model_end"

    REQUEST_ERROR = "error"
    REQUEST_TOOL_END = "tool_end"


## REQUEST LIFECYCLE
class RequestConnect(BaseModel):
    type: DeltaType= DeltaType.REQUEST_START
    request_id: str
    status: str = "running"
    pong: float = Field(default_factory=lambda: datetime.utcnow().timestamp())

class ContentModeration(BaseModel):
    type: DeltaType= DeltaType.REQUEST_MODERATED
    request_id: str
    moderate: None | dict[str, Any] = None


## CHUNK STREAMING
class ChunkStart(BaseModel):
    type: DeltaType= DeltaType.REQUEST_CHUNK_START
    run_id: str
    graph_node: dict[str, Any] = {}
    params: dict[str, Any] = {}

class ChunkMessage(BaseModel):
    type: DeltaType= DeltaType.REQUEST_CHUNK
    run_id: str
    parts: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    response_metadata: dict[str, Any] = {}
    usage_metadata: dict[str, Any] = {}

class ChunkEnd(BaseModel):
    type: DeltaType= DeltaType.REQUEST_CHUNK_END
    run_id: str
    response_metadata: dict[str, Any] = {}


class ChunkToolEnd(BaseModel):
    type: DeltaType= DeltaType.REQUEST_TOOL_END
    run_id: str
    tool_id: str
    tool_name: str
    data: dict[str, Any] = {}

## END OF STREAM
class ErrorResponse(BaseModel):
    type: DeltaType= DeltaType.REQUEST_ERROR
    # Stable machine-readable tag — frontend switches on this:
    #   auth_error | rate_limit | bad_request | network_error | upstream_error | internal_error
    error_type: str = "internal_error"
    # Human-readable message (safe to display).
    error: str
    # Upstream HTTP status when the failure came from an external API; null otherwise.
    code: int | None = None

class RequestEnd(BaseModel):
    type: DeltaType= DeltaType.REQUEST_END
    request_id: str
    status: str = "completed"
    total_time: float = 0.0
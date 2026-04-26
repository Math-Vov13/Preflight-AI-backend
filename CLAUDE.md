# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Stack and entry point

FastAPI + LlamaIndex Workflow backend, Python 3.13 (pinned in `pyproject.toml`), managed with `uv`. **Two distinct LLM paths**: (a) the **chat workflow** (`src/rag/config.py` → `ChatWorkflow` in `src/rag/server.py`) uses **Google Gemini** via `llama-index-llms-google-genai` (model from `GEMINI_MODEL`, default `gemini-2.5-flash`, key from `GOOGLE_API_KEY`); (b) the **preflight pipeline** + the **`tools_passthrough=True` route** in `/generation/` use **SiliconFlow** via `SiliconFlowClient` (`src/models/siliconflow.py`, OpenAI-compatible, key from `SILICONFLOW_API_KEY`, per-task model env vars `ONTOLOGY_MODEL` / `PERSONA_MODEL` / `SIMULATION_MODEL` / `REPORT_MODEL` / `JUDGE_MODEL` / `CHAT_MODEL`). When swapping providers, touch only the path that's relevant — they're independent. Four external stores: Redis (chat history), Postgres/Supabase (relational), Qdrant (vectors), and blob storage — historically AWS S3 (`src/models/s3/`, used by `/collections`) and Supabase Storage (`src/models/supabase_storage/`, used by `/generation` chat attachments).

`README.md` is **out of date** (it describes a LangChain + Gemini app). `README2.md` is the accurate one — read that for the full system architecture (frontend included).

## Common commands

```bash
uv sync                    # install deps from uv.lock (Python 3.13 required)
uv run src/server.py       # dev server on :8080 (FastAPI mounted at root_path=/api/v1)
uv lock                    # refresh lockfile after editing pyproject.toml
docker build -t preflight-ai-backend .
```

There is no test suite, no linter config, and no `Makefile`. `main.py` at the repo root is a stub — do not use it.

The `Dockerfile` final stage runs the app as a non-root `appuser` (UID/GID 1001, home `/app`). The `.venv` is created at container start by `uv run` (the builder-stage venv copy is intentionally commented out), so `appuser` needs write access to `/app` — don't drop the `--chown` on the `COPY` lines.

## Run from the repo root, always

Two things break when run from elsewhere:

1. `src/rag/server.py` reads system prompts via **relative** paths (`open("src/docs/GEMINI_SYSTEM_PROMPT.md")`) at import time.
2. `src/server.py` uses bare imports (`from endpoints...`, `from models...`, `from rag...`). These resolve only because launching `uv run src/server.py` adds `src/` to `sys.path`. Do not switch to `python -m src.server` or `uvicorn src.server:app` without converting the imports to package-relative form.

## Architecture

```
HTTP → src/server.py (FastAPI, root_path=/api/v1)
        ├─ /generation/   → endpoints/generation.py  (SSE chat stream + orchestrator passthrough)
        ├─ /collections/* → endpoints/collections.py (RAG ingestion)
        ├─ /runs/new      → endpoints/control.py     (preflight pipeline trigger — legacy + orchestrated)
        ├─ /runs/{id}     → endpoints/runs.py        (assembled artefacts + metrics)
        ├─ /stream        → endpoints/stream.py      (per-user SSE event bus)
        └─ /health        → pings Redis, Postgres, Qdrant (does NOT check Supabase Storage)
```

A request middleware in `src/server.py` tags every request with a uuid and logs `METHOD path | client=…` then `status=… elapsed=…s` to both stderr and `app.log` (file handler at the repo root).

### Chat flow (`endpoints/generation.py` → `rag/server.py`)

1. Load prior turns from Redis (`models/cache_redis/chat_history.py`, 7-day TTL, JSON, **text-only** — multimodal blocks are stripped before persisting; `additional_kwargs` is kept and currently carries `attachments` metadata). Roles are stored as the enum **value** (`"user"`, `"assistant"`); the deserializer also accepts the legacy `"MessageRole.USER"` form left in old cache entries.
2. If `request.files` is set, decode each `base64` payload (raw or `data:` URL) and upload to Supabase Storage via `models/supabase_storage/upload_files.upload_file_to_supabase` in parallel (`asyncio.to_thread`). Best-effort: a missing/unconfigured client returns `None` and the chat still proceeds. Returned object keys are stashed on the persisted user message as `additional_kwargs["attachments"] = [{name, mime_type, size, storage_key}, ...]`. The LLM message itself still inlines the original base64 as `ImageBlock(url=...)` so the model can see the content.
3. Build a `ChatWorkflow` (LlamaIndex `Workflow`) with three steps that loop until the LLM stops calling tools:
   - `retrieve` — if `collection_name` is set, embed the last user message via `openai_ef` (OpenAI `text-embedding-3-small`, 1536 dims), query Qdrant top-5 (cosine distance), splice docs into `USER_MESSAGE_WITH_DOCS_CONTEXT.md` template. Document text is stored in each point's `payload["document"]`.
   - `generate` — `astream_chat_with_tools`, emit `LLMTurnStart` / `LLMTextDelta` / `LLMToolCallsAnnounced` / `LLMTurnEnd` to the SSE stream.
   - `call_tools` — execute requested tools, append `ChatMessage(role=TOOL)` results, loop back to `generate`. Tools live in `rag/tools/`: `web_search` (Tavily), `get_tle` + `get_satellite_position` (N2YO), `code_interpreter` (Jupyter), `generate_image` (SiliconFlow FLUX).
4. After the workflow finishes, persist the user message + final assistant message back to Redis.

The endpoint hardcodes `collection_name="1234"` (`endpoints/generation.py:172`, marked `TODO: per-user collection`) — there is no per-user RAG collection yet.

### SSE event names (matters for the frontend proxy)

`generation.py` emits these `event:` names — the Next.js proxy at `api-client/chat` rewrites them into a different schema. Do not rename these without updating the frontend:

- `delta` — used for `ChunkStart`, `ChunkMessage` (text + tool_calls), `ChunkEnd`
- `(default)` — `ContentModeration`, `ChunkToolEnd`, `ErrorResponse`
- terminator: `data: [DONE]\n\n`

### Orchestrator passthrough (`/generation/` extension)

The chat-completions route now accepts four extra `GenerationRequest` fields used by the **preflight orchestrator** on the FE:

```py
system: str | None              # custom system prompt overriding the workflow default
tools: list[dict] | None        # OpenAI- or Anthropic-style tool definitions
tool_choice: dict | str | None  # {type: "tool", name: "..."} | "auto" | "required"
tools_passthrough: bool = False # when true: bypass workflow, single LLM turn, NO tool execution
```

When `tools_passthrough=True` the route **skips the LlamaIndex `ChatWorkflow` entirely** and calls `SiliconFlowClient.chat_stream_with_tools` directly. It streams text deltas as `chat_model_stream` events and emits one terminal `chat_model_stream` carrying the assembled `tool_calls` once the model finishes. The advertised tool is **never executed downstream** — it's used purely as a structured-output channel (the FE consumes `tool_calls[0].args` as a typed JSON payload). `_normalize_tools` and `_normalize_tool_choice` coerce Anthropic-style `{name, description, input_schema}` and `{type:"tool", name}` to the OpenAI shape SiliconFlow expects, so the FE can stay model-agnostic.

This is what the FE preflight route uses to extract the orchestration JSON from the chat LLM (forced `tool_choice: { type: "tool", name: "submit_orchestration" }`).

### Preflight pipeline (legacy + orchestrated)

`POST /api/v1/runs/new` (`endpoints/control.py`) triggers a preflight run. Two modes, branched in `_do_run`:

1. **Legacy / hardcoded 5-phase** (`services/pipeline.run_full_pipeline`) — ontology → panel → simulation → validation → judge. Used when `req.steps is None`. `panel_size` (3–50) and `rounds` (1–3) are tunable, and `model_overrides` lets the caller swap models per phase. Persists artefacts to disk + Postgres + Zep.

2. **Orchestrated / dynamic-steps** (`services/orchestrated_pipeline.run_orchestrated_pipeline`) — used when `req.steps` is present. The FE chat LLM emits a `steps[]` JSON via the orchestrator passthrough above; the executor runs them in order, resolving dependencies (`target_step_id` for review, `connected_steps_id` for judge), and emits **generic `step.*` events** the FE reducer keys by `step_id`:

   ```
   step.start  { run_id, step_id, step_type, name }
   step.update { run_id, step_id, step_type, set?, append?, payload?, details? }
   step.done   { run_id, step_id, step_type, latency_s, summary?, payload }
   step.error  { run_id, step_id, step_type, error }
   ```

   Step types: `panel` (10 personas, server-fixed — orchestrator can't tune count or models), `simulation` (OASIS forum, 2 rounds, with the `metadata.goal` injected into the brief), `review` (direct LLM call with `metadata.system_prompt` against the targeted simulation's forum), `judge` (synthesises verdict + scores from `metadata.connected_steps_id` payloads via the judge model). `validate_steps` enforces structural + semantic rules synchronously in the route handler so a malformed payload 400s instead of producing a 30s `run.error`.

   Per-step artefact persistence to `run_artifacts` is **not yet wired** because the `run_artefact_kind` enum doesn't accept `step:<id>` keys — see `persist_orchestrated_run`'s comment for the migration to ship before turning that on. Until then, artefacts live only in the SSE event log and the `runs` row gets the terminal verdict + cost.

   Heartbeats (`run.heartbeat` every 10s) and idempotency (`Idempotency-Key` header, 5-min TTL) work identically in both modes.

### Error mapping

`_classify_error` in `endpoints/generation.py` maps `openai.*` exceptions to a stable `ErrorResponse` schema (`error_type`, `error`, `code`). It still covers the SiliconFlow `tools_passthrough` path, but **Gemini errors from the chat workflow are not classified** — they hit the generic `internal_error` branch. When adding LLM error paths (or porting the chat workflow's `google.genai` exceptions), extend this function rather than raising new shapes; the auth-error message also still names `SILICONFLOW_*` env vars and is misleading on the Gemini path.

### Storage clients

All clients in `src/models/*/client.py` are module-level singletons created at import time:

- `cache_redis/client.py` — exposes both standard Redis (`client`, `async_client` from `REDIS_URL`) and Upstash REST (`upstash_client`, `async_upstash_client` from `UPSTASH_REDIS_REST_URL` + `_TOKEN`). Each is `None` if its credentials are missing. Pings on import — non-fatal failures still let the app start, but tracebacks will appear in logs.
- `pgsql/client.py` — `psycopg` sync connection (autocommit). Used only by `/health`; not by request handlers.
- `vc_qdrant/client.py` — sync `QdrantClient` from `QDRANT_URL` (+ optional `QDRANT_API_KEY`). The async constructor exists (`create_connection` → `AsyncQdrantClient`) but isn't used. `vc_qdrant/utils.py` provides `openai_ef` (raw OpenAI embeddings call) plus collection helpers (`ensure_collection`, `add_documents`, `query_documents`, `delete_items`, `get_collection_items`, `count_collection`, `list_collection_names`); collections are created with 1536-dim cosine vectors on first use.
- `s3/upload_files.py` — `boto3` client built on demand (no module-level singleton). Bucket name comes from `CONFIG.S3_BUCKET_NAME` (`src/config/config.py`, hardcoded `"the-universal-agent"`). Object keys are namespaced under `user_upload/` or `generated/`. Used by `/collections` ingestion only.
- `supabase_storage/client.py` — `supabase.Client` from `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` (service role required to bypass RLS server-side); `None` if either is missing. `upload_files.upload_file_to_supabase` writes to `CONFIG.SUPABASE_STORAGE_BUCKET` (default `chat-uploads`, overridable via env) under `user_upload/{kind}/...` and returns the object path; `create_signed_url` mints time-limited links.

## Environment

`.env` at the repo root is loaded by every module that calls `load_dotenv()`. `.env.example` is the template. **LLM keys**: `GOOGLE_API_KEY` powers the chat workflow (model selectable via `GEMINI_MODEL`, default `gemini-2.5-flash`); `SILICONFLOW_API_KEY` powers the preflight pipeline and the `tools_passthrough` route (per-task models via `ONTOLOGY_MODEL` / `PERSONA_MODEL` / `SIMULATION_MODEL` / `REPORT_MODEL` / `JUDGE_MODEL` / `CHAT_MODEL`, plus optional `SILICONFLOW_BASE_URL` to switch the `.com` international endpoint to `.cn`). `REDIS_URL` accepts both `redis://` and `rediss://` (Upstash TLS). For Upstash REST clients, set `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN`. `QDRANT_URL` defaults to `http://localhost:6333`; set `QDRANT_API_KEY` for Qdrant Cloud. For Supabase Storage, set `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` (service role, **not** anon — anon will be blocked by RLS) and optionally `SUPABASE_STORAGE_BUCKET` (default `chat-uploads`, must be the bucket **name** like `chat-uploads`, not the storage endpoint URL). For S3 uploads via `/collections`, `boto3` reads `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` from the standard chain. The backend is normally consumed via the frontend in the parent `docker-compose.yml`, where compose overrides `REDIS_URL` / `QDRANT_URL` to internal service names.

The repo's `.env` is shared with the Next.js frontend. The backend reads `SUPABASE_URL` only — the `NEXT_PUBLIC_SUPABASE_URL` / `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` entries are frontend-only and are ignored by the Python clients. If only `NEXT_PUBLIC_SUPABASE_URL` is set, `supabase_storage/client.py` silently disables the storage client and chat uploads become no-ops.

## Editable runtime assets

`src/docs/GEMINI_SYSTEM_PROMPT.md` and `src/docs/USER_MESSAGE_WITH_DOCS_CONTEXT.md` are read **at module import** of `rag/server.py`. Edits take effect on the next process start, not on every request — restart the server after changing them.

## Gotchas

### openai pydantic objects in SSE event payloads

The `openai` package's `BaseModel` is configured with `defer_build=True`, so its model classes carry a `MockValSer` placeholder serializer until forced. During streaming, `llama-index-llms-openai` stores raw `ChoiceDeltaToolCall` objects in `assistant_msg.additional_kwargs["tool_calls"]`. If those objects end up inside a `dict[str, Any]` field of one of our own pydantic SSE models (`ChunkEnd.response_metadata`, `ChunkMessage.tool_calls`, etc.), `model_dump_json()` raises `PydanticSerializationError: 'MockValSer' object is not an instance of 'SchemaSerializer'` mid-stream — the symptom is "the tool call is announced to the frontend but no further events arrive".

`rag/server.py` exposes `_jsonable(obj)` which recursively converts nested `pydantic.BaseModel` (including openai's, via its `to_dict(mode="json")`) into plain dicts/lists. Run any value sourced from llama_index `additional_kwargs` / `ChatResponse` through `_jsonable` before assigning it to one of our event models.

### Stream errors leave the workflow running

Errors raised inside `event_stream()` in `endpoints/generation.py` (e.g. JSON serialization) are caught, mapped to `ErrorResponse`, and followed by `[DONE]`. The LlamaIndex `WorkflowHandler` is **not** cancelled — it keeps executing tools and making further LLM calls in the background after the client has disconnected. Phantom Gemini / SiliconFlow requests in the logs after an error are usually this, not a separate request.

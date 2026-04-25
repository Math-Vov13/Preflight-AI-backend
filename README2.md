# Preflight AI — Universal Agent RAG

Full-stack RAG chat application with streaming, tool use, and per-conversation file uploads. Next.js 15 frontend + FastAPI / LlamaIndex Workflow backend, backed by Supabase Postgres, Redis, and Qdrant.

![Architecture](./references/agent_portait.png)

## Stack

| Layer | Tech |
|---|---|
| Frontend | Next.js 15 (App Router, React 19, TS), Tailwind, Zustand, Drizzle ORM |
| Backend | FastAPI, LlamaIndex Workflow, Python 3.13 |
| LLM | SiliconFlow (OpenAI-compatible) — default `zai-org/GLM-4.5-Air` |
| Auth + DB | Supabase (Auth + Postgres via Transaction pooler) |
| Cache | Redis Stack (chat history JSON, 7-day TTL) |
| Vector store | Qdrant (RAG, embeddings via OpenAI `openai_ef`, cosine 1536-dim) |
| Tools | Tavily web search, satellite TLE / position (N2YO), Jupyter sandbox, SiliconFlow FLUX.2 image gen |

## Architecture

```
┌──────────────┐      ┌──────────────┐      ┌─────────────────────┐
│  Browser     │──────▶  Next.js     │──────▶  FastAPI (app:8080) │
│              │      │  (port 3000) │      │  LlamaIndex Workflow│
└──────────────┘      └──────────────┘      └─────────┬───────────┘
                                                       │
                          ┌────────────────────────────┼─────────────────────────┐
                          ▼                            ▼                         ▼
                  ┌────────────────┐          ┌────────────────┐         ┌────────────────┐
                  │  Supabase PG   │          │  Redis (cache) │         │  Qdrant (RAG)  │
                  │  (auth + data) │          │   port 6379    │         │   port 6333    │
                  └────────────────┘          └────────────────┘         └────────────────┘
```

NGINX (`nginx.conf`) is committed but **commented out** in `docker-compose.yml`; the `web` service talks to `app` directly. Re-enable nginx if you need its streaming-tuned proxy config.

## Quick start (Docker)

```bash
git clone <repo-url> preflight-ai && cd preflight-ai

# 1. Single root .env — copy templates and fill in keys
cp web-frontend/.env.example .env       # then merge backend keys in:
cat app-backend/.env.example >> .env    # dedupe by hand if needed

# 2. Build and run
docker-compose up --build
```

Services come up on:
- Web UI — http://localhost:3000
- FastAPI (internal) — `app:8080` (not exposed; reach via the web app or temporarily map a port)
- Redis — `localhost:6379`
- Qdrant — internal only

The `db` service is commented out — the project uses a hosted Supabase Postgres. **Schema changes must be pushed to Supabase manually** (`pnpm migrate:push` from `web-frontend/`, or run `init.sql` in the Supabase SQL editor).

## Local development (without Docker)

Symlink the root `.env` so each app picks it up natively:

```bash
ln -s ../.env web-frontend/.env.local
ln -s ../.env app-backend/.env
```

### Frontend

```bash
cd web-frontend
pnpm install
pnpm dev                # Turbopack dev server on :3000
pnpm lint
pnpm build

# Drizzle migrations (against Supabase via DATABASE_URL)
pnpm migrate            # generate SQL files into ./migrations/
pnpm migrate:push       # apply to the connected DB
pnpm migrate:docker     # export schema → ../init.sql (Docker init seed)
```

### Backend

```bash
cd app-backend
uv sync                 # Python 3.13 required (pinned in pyproject.toml)
uv run src/server.py    # dev server on :8080
```

## Environment variables

A **single root `.env`** is the source of truth — `docker-compose.yml` consumes it for both services. Required keys:

```env
# Supabase
DATABASE_URL=postgresql://postgres.<ref>:<pwd>@aws-0-<region>.pooler.supabase.com:6543/postgres
NEXT_PUBLIC_SUPABASE_URL=https://<ref>.supabase.co
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=sb_publishable_...

# LLM / tools
SILICONFLOW_API_KEY=         # required (chat + image gen)
SILICONFLOW_MODEL=           # optional, default zai-org/GLM-4.5-Air
SILICONFLOW_BASE_URL=        # optional, default https://api.siliconflow.com/v1
OPENAI_API_KEY=              # OpenAI embedding function (text-embedding-3-small)
TAVILY_API_KEY=              # web_search tool
N2YO_API_KEY=                # satellite tools
AWS_ACCESS_KEY_ID=           # S3 ingestion bucket
AWS_SECRET_ACCESS_KEY=

# CDN (frontend, baked at build)
NEXT_PUBLIC_CLOUDFRONT=
NEXT_GEN_AI_CDN=

# Backend infra (only used outside Docker — compose overrides these)
REDIS_URL=redis://localhost:6379
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=               # required for Qdrant Cloud, blank for local
```

> `NEXT_PUBLIC_*` is baked at build time — rebuild the `web` image after any change.

## API endpoints

Backend (FastAPI, `root_path=/api/v1`):

| Method | Path | Description |
|---|---|---|
| POST | `/generation/` | SSE stream — runs the LlamaIndex `ChatWorkflow` end-to-end |
| GET  | `/collections/` | List user collections |
| GET  | `/collections/{id}/items` | List items in a collection |
| POST | `/collections/{id}/items` | Upload doc (txt/json/pdf), chunk, embed, index in Qdrant |
| DELETE | `/collections/{id}/items/{item_id}` | Remove an item |
| GET  | `/health` | Pings Redis, Postgres (`SELECT 1`), Qdrant `get_collections` |

Frontend route handlers (`/api-client/*`):

| Path | Description |
|---|---|
| `chat` | SSE proxy that transforms backend `DeltaType` events into the simplified client stream |
| `createChat`, `history` | Conversation lifecycle |
| `auth/login`, `auth/signup`, `auth/logout` | Supabase Auth handlers |
| `proxy/fetch`, `proxy/redirect` | Server-side link-preview helpers (token-gated) |

The middleware rewrites `POST /login` → `/api-client/auth/login` and `POST /signup` → `/api-client/auth/signup` so the natural URLs work as both pages (GET) and API endpoints (POST).

## Project layout

```
.
├── docker-compose.yml          # web + app + cache + vectordb (nginx, db commented out)
├── nginx.conf                  # streaming-tuned proxy config (kept for re-enable)
├── init.sql                    # Drizzle-generated schema + stored procedures (Supabase seed)
├── .env                        # single source of truth (gitignored)
├── web-frontend/
│   └── src/
│       ├── app/
│       │   ├── (client)/chat/[conv_id]/        # active conversation
│       │   ├── (server)/api-client/            # route handlers (chat proxy, auth, history)
│       │   ├── login/, signup/, pricing/
│       │   └── middleware.ts                   # Supabase session refresh + auth gate
│       ├── components/Providers/historyProvider.tsx   # Zustand store (sendMessage, upsertTool)
│       ├── db/                                 # Drizzle schema + stored-proc callers
│       └── lib/
│           ├── supabase/                       # browser + server clients
│           └── types/{db,client}.schema.ts     # two distinct Zod schemas — do not merge
└── app-backend/
    └── src/
        ├── server.py                # FastAPI entry, root_path=/api/v1
        ├── endpoints/{generation,collections}.py
        ├── rag/
        │   ├── server.py            # ChatWorkflow (retrieve → generate → call_tools)
        │   ├── config.py            # OpenAILike client (SiliconFlow)
        │   └── tools/               # web_search, satellite, code_interp, image_gen
        ├── models/
        │   ├── cache_redis/chat_history.py     # text-only Redis JSON store
        │   └── vc_qdrant/                      # Qdrant client + collection helpers
        └── docs/GEMINI_SYSTEM_PROMPT.md        # loaded at runtime — edit takes effect immediately
```

## Streaming protocol (SSE)

Two distinct event shapes — the proxy at `api-client/chat` rewrites one into the other:

| Backend event (`DeltaType`) | Proxy → Client event |
|---|---|
| `content_moderation` | (consumed) |
| `chat_model_start/stream/end` | `delta` (text chunks) |
| `tool_end` | `tool_end` (merged with prior `tool_start` by `call_id`) |
| `error` (`{error_type, error, code}`) | `error` (`{error, err_type, code}`) |

The proxy injects synthetic `startup`, `summary`, `limits`, and `session` events that don't exist on the backend. Multi-stage tool tracking: a `stage` counter increments on each `chat_model_start`, producing tool paths like `generation_task:tools:2/0`.

## Pricing tiers (frontend)

`Free` · `Pro $19/mo` · `Operator $49/mo` — defined on the `/pricing` page and stored in `users.plan`. **Subscription enforcement is not wired yet**; the chat proxy currently sends a static `"pro"` placeholder.

## Status

| Area | State |
|---|---|
| Frontend persistence (Drizzle + Supabase stored procs) | ✅ |
| Supabase Auth (login, signup, middleware) | ✅ |
| RAG ingestion (`/collections/{id}/items` → S3 → Qdrant) | ✅ |
| Streaming chat with tool use | ✅ |
| Per-user Qdrant collection | ❌ hardcoded `"1234"` |
| Plan enforcement | ❌ static `"pro"` placeholder |
| Reasoning toggle | ❌ local UI state only |

## License

MIT — see `LICENSE`.

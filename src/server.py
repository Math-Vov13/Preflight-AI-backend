import asyncio
import logging
import os
import platform
from contextlib import asynccontextmanager
from time import time
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware

from endpoints.collections import router as collections_router
from endpoints.generation import router as generation_router
from models import preflight_db
from models.cache_redis.client import client as redis_client
from models.pgsql.client import client as pgsql_client
from models.vc_qdrant.client import qdrant_client

# Ported from /preflight/backend — multi-agent simulation + run
# orchestration + Zep cross-run memory + Supabase JWT auth.
from endpoints.preflight_auth import router as preflight_auth_router
from endpoints.briefs import router as briefs_router
from endpoints.chat import router as chat_router
from endpoints.control import router as control_router
from endpoints.graph import router as graph_router
from endpoints.runs import router as runs_router
from endpoints.stream import router as stream_router
from events import attach_loop


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ported event-bus needs the running loop registered so worker threads
    # (the simulation pipeline runs in a thread pool) can schedule SSE
    # event deliveries from outside async context.
    attach_loop(asyncio.get_running_loop())
    # Surface preflight DB readiness loud and early. If DATABASE_URL is
    # set but the Drizzle migrations haven't been run, every endpoint
    # silently falls back to file mode — no warning, no broken response,
    # just no persistence. Log clearly so the misconfiguration is caught
    # before users start runs.
    schema = preflight_db.schema_check()
    if schema["database_url_set"] and not schema["schema_ok"]:
        missing = [
            t for t, present in schema["tables_present"].items() if not present
        ]
        logging.warning(
            "preflight: DATABASE_URL is set but schema is incomplete "
            "(missing: %s, connection_ok=%s). Falling back to file-mode "
            "persistence. Run drizzle migrations on the frontend project to fix.",
            ", ".join(missing) or "none",
            schema["connection_ok"],
        )
    elif schema["schema_ok"]:
        logging.info("preflight: DB schema present, persistence enabled")
    # Recover any runs left in status='running' from a previous process
    # — a crash mid-pipeline would otherwise lock the owner out of new
    # runs forever (per-user concurrency in endpoints/control.py).
    recovered = preflight_db.recover_orphan_runs()
    if recovered:
        logging.info("preflight: recovered %d orphan run(s) on startup", recovered)
    yield


app = FastAPI(root_path="/api/v1", lifespan=lifespan)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid4())
    start = time()
    client_host = request.client.host if request.client else "unknown"
    logging.info(
        f"[{request_id}] {request.method} {request.url.path} | client={client_host}"
    )
    try:
        response = await call_next(request)
    except Exception as exc:
        logging.error(f"[{request_id}] exception: {exc} | {time() - start:.3f}s")
        raise
    elapsed = time() - start
    log = logging.warning if response.status_code >= 400 else logging.info
    log(f"[{request_id}] status={response.status_code} elapsed={elapsed:.3f}s")
    return response


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/health")
def health_check():
    pgsql_status = False
    if pgsql_client:
        try:
            pgsql_client.execute("SELECT 1")
            pgsql_status = True
        except Exception:
            pgsql_status = False

    qdrant_status = False
    if qdrant_client:
        try:
            qdrant_client.get_collections()
            qdrant_status = True
        except Exception:
            qdrant_status = False

    redis_status = False
    if redis_client:
        try:
            redis_status = bool(redis_client.ping())
        except Exception:
            redis_status = False

    return {
        "status": "healthy",
        "redis": redis_status,
        "pgsql": pgsql_status,
        "qdrant": qdrant_status,
        # Preflight-specific persistence story: distinct from the generic
        # `pgsql` ping above because preflight uses its own connection per
        # call and cares about which tables are reachable, not just that
        # SELECT 1 worked.
        "preflight": preflight_db.schema_check(),
    }


# CORS — `allow_origins=["*"]` together with `allow_credentials=True`
# violates the CORS spec; browsers reject credentialed responses with
# a wildcard origin. Read an explicit allowlist from CORS_ALLOW_ORIGINS
# (comma-separated). Empty/unset falls back to a permissive dev default
# AND drops allow_credentials so cookie/Authorization-credentialed
# requests still work without browser hostility.
def _parse_cors_origins(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [o.strip() for o in raw.split(",") if o.strip()]


_cors_origins = _parse_cors_origins(os.getenv("CORS_ALLOW_ORIGINS"))
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )
else:
    logging.warning(
        "preflight: CORS_ALLOW_ORIGINS unset — falling back to wildcard "
        "without credentials. Set it to a comma-separated origin list "
        "(e.g. https://app.preflight.dev,http://localhost:3000) before "
        "deploying so credentialed requests work in prod.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )
app.include_router(generation_router, prefix="/generation", tags=["generation"])
app.include_router(collections_router, prefix="/collections", tags=["collections"])

# Ported routes — auth + run lifecycle + chat-on-run + brief parsing + Zep
# graph search + SSE stream. Auth: header-based for the JSON endpoints
# (Supabase JWT via auth.CurrentUser), query-param-based for /stream
# (EventSource can't set headers; see endpoints/stream.py).
app.include_router(preflight_auth_router)  # /auth/whoami, /auth/mode
app.include_router(control_router)         # /runs/new
app.include_router(runs_router)            # /runs, /runs/{id}
app.include_router(chat_router)            # /runs/{id}/chat (POST + GET)
app.include_router(briefs_router)          # /briefs/parse
app.include_router(graph_router)           # /graph/search, /graph/status
app.include_router(stream_router)          # /stream


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

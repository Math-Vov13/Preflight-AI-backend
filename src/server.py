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
from models.cache_redis.client import client as redis_client
from models.pgsql.client import client as pgsql_client
from models.vc_qdrant.client import qdrant_client


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    }


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(generation_router, prefix="/generation", tags=["generation"])
app.include_router(collections_router, prefix="/collections", tags=["collections"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

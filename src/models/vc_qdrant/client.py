from os import environ as env

from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient, QdrantClient

load_dotenv()

_url = env.get("QDRANT_URL", "http://localhost:6333")
_api_key = env.get("QDRANT_API_KEY") or None

qdrant_client = QdrantClient(url=_url, api_key=_api_key)


async def create_connection() -> AsyncQdrantClient:
    return AsyncQdrantClient(url=_url, api_key=_api_key)


try:
    print("Qdrant ping:", len(qdrant_client.get_collections().collections), "collections")
except Exception as exc:
    print("Qdrant ping failed:", exc)

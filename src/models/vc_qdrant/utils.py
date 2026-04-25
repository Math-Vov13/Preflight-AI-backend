from os import environ as env
from typing import Any

from openai import OpenAI
from qdrant_client.http.models import (
    Distance,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from models.vc_qdrant.client import qdrant_client

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536

_openai_client = OpenAI(api_key=env.get("OPENAI_API_KEY"))


def openai_ef(input: str | list[str]) -> list[list[float]]:
    """Embed text(s) with OpenAI. Mirrors Chroma's EmbeddingFunction callable surface."""
    texts = [input] if isinstance(input, str) else list(input)
    if not texts:
        return []
    response = _openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def collection_exists(name: str) -> bool:
    try:
        return qdrant_client.collection_exists(collection_name=name)
    except Exception as exc:
        print(f"Error checking collection '{name}':", exc)
        return False


def ensure_collection(name: str) -> bool:
    """Create the collection with cosine-distance 1536-dim vectors if it doesn't exist."""
    try:
        if not qdrant_client.collection_exists(collection_name=name):
            qdrant_client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=EMBEDDING_DIMS, distance=Distance.COSINE),
            )
        return True
    except Exception as exc:
        print(f"Error creating collection '{name}':", exc)
        return False


def list_collection_names() -> list[str]:
    return [c.name for c in qdrant_client.get_collections().collections]


def add_documents(
    collection_name: str,
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
) -> None:
    points = [
        PointStruct(id=pid, vector=vec, payload={**meta, "document": doc})
        for pid, doc, vec, meta in zip(ids, documents, embeddings, metadatas)
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)


def get_collection_items(collection_name: str) -> dict[str, Any]:
    """Return all stored points in a Chroma-like shape: {ids, documents, metadatas}."""
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    next_page = None
    while True:
        points, next_page = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=next_page,
        )
        for point in points:
            payload = dict(point.payload or {})
            documents.append(payload.pop("document", ""))
            metadatas.append(payload)
            ids.append(str(point.id))
        if next_page is None:
            break

    return {"ids": ids, "documents": documents, "metadatas": metadatas}


def count_collection(collection_name: str) -> int:
    return qdrant_client.count(collection_name=collection_name, exact=True).count


def delete_items(collection_name: str, ids: list[str]) -> None:
    qdrant_client.delete(
        collection_name=collection_name,
        points_selector=PointIdsList(points=ids),
    )


def query_documents(collection_name: str, query_embedding: list[float], limit: int = 5) -> list[str]:
    """Return the top-k document strings for a precomputed embedding."""
    result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=limit,
        with_payload=True,
    )
    return [
        (point.payload or {}).get("document", "")
        for point in result.points
        if point.payload
    ]

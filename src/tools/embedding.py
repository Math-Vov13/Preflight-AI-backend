"""Embedding wrapper over the SiliconFlow client."""
from __future__ import annotations

from sim_config import settings
from models.siliconflow import client


def embed_batch(texts: list[str], *, model: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    m = model or settings().embedding_model
    return client().embed(texts, model=m)

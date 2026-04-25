"""Diversity score over a list of text contributions via embedding distance."""
from __future__ import annotations

import math

from tools.embedding import embed_batch


def mean_pairwise_cosine_distance(texts: list[str]) -> float:
    """Average (1 - cos_sim) across all unique pairs. Range ~ [0, 2].

    Higher = more diverse. Returns 0.0 for < 2 items.
    """
    if len(texts) < 2:
        return 0.0
    embeddings = embed_batch(texts)
    if not embeddings:
        return 0.0

    normalized: list[list[float]] = []
    for vec in embeddings:
        norm = math.sqrt(sum(x * x for x in vec))
        normalized.append(vec if norm == 0 else [x / norm for x in vec])

    total = 0.0
    pairs = 0
    n = len(normalized)
    for i in range(n):
        for j in range(i + 1, n):
            dot = sum(a * b for a, b in zip(normalized[i], normalized[j]))
            total += 1.0 - dot
            pairs += 1
    return total / pairs if pairs else 0.0

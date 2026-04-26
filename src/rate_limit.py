"""Per-user in-process token bucket — light DoS protection for expensive
endpoints (PDF parsing, etc.).

Single-process only: the bucket lives in memory, so multiple replicas
each enforce their own quota. For hackathon scale that's fine. Swap to
Redis-based throttling before scaling out.

Usage:

    from rate_limit import TokenBucket

    _BRIEFS_BUCKET = TokenBucket(capacity=10, refill_per_sec=10/60)

    @router.post("/briefs/parse")
    def parse(user: CurrentUser):
        retry_after = _BRIEFS_BUCKET.try_consume(user)
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail="rate limited",
                headers={"Retry-After": str(retry_after)},
            )
        ...
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class TokenBucket:
    """Per-key token bucket. `try_consume(key)` returns None on success
    or the recommended Retry-After (in seconds, ceil-rounded) on rejection."""

    def __init__(self, *, capacity: int, refill_per_sec: float) -> None:
        self.capacity = float(capacity)
        self.refill_per_sec = float(refill_per_sec)
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def try_consume(self, key: str, cost: float = 1.0) -> int | None:
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=self.capacity, last_refill=now)
                self._buckets[key] = bucket
            # Refill since last check
            elapsed = now - bucket.last_refill
            bucket.tokens = min(
                self.capacity, bucket.tokens + elapsed * self.refill_per_sec,
            )
            bucket.last_refill = now
            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return None
            # Out of tokens — recommend a wait long enough for `cost` to refill.
            need = cost - bucket.tokens
            retry_after_s = need / self.refill_per_sec if self.refill_per_sec > 0 else 60
            return max(1, int(retry_after_s) + 1)

    def reset(self, key: str | None = None) -> None:
        """Test helper. `key=None` clears every bucket."""
        with self._lock:
            if key is None:
                self._buckets.clear()
            else:
                self._buckets.pop(key, None)

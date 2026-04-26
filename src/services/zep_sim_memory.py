"""Per-persona Zep thread memory for the OASIS simulation phase.

Why this exists (separate from `services/zep_memory.py`):
* `zep_memory.ZepMemory` ingests *post-run artefacts* into the user's knowledge
  graph (entities, verdicts, validation report). Cross-run, semantic, queried
  much later via `graph.search`.
* `ZepSimMemory` (this module) is *intra-run scratchpad memory*: each persona
  gets their own Zep thread for the duration of one run, holds round-by-round
  forum activity, and feeds back a compressed `get_user_context()` summary at
  the start of every round. This is what fixes "the agent forgets between
  rounds" — Camel's local memory still grows, but it gets a Zep-summarised
  recap injected as a system note so prior stance stays anchored even when
  Camel's context window starts truncating older turns.

Scoping
-------
One Zep user per persona-per-run: ``pf-sim-<run_id>-<persona_id>``. One thread
per persona-per-run: ``pf-thread-<run_id>-<persona_id>``. Run-scoping prevents
forum chatter from one product brief leaking into another via Zep retrieval —
that's a job for the post-run graph ingestion, not the simulation memory.

Failure policy
--------------
Best-effort. Every external call is wrapped — exceptions are logged and a
falsy value is returned. The simulation must never fail because Zep is down,
slow, or rate-limited. Speed of the run also wins over completeness: writes
fire as background tasks; only the read path (``get_persona_context``) is
awaited because the caller actually needs the string.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import Iterable

from zep_cloud import AsyncZep, Message
from zep_cloud.core.api_error import ApiError

from sim_config import settings
from schemas.persona import Persona
from schemas.scenario import ForumComment, ForumPost

logger = logging.getLogger(__name__)


# Hard cap on the number of messages we push per persona per run. A 20-persona,
# 3-round forum produces roughly 1 post + ~3 comments per round per persona =
# ~12 messages each — well under the cap. The limit is a guardrail against
# pathological loops, not normal operation.
_MAX_MESSAGES_PER_PERSONA = 60

# Cap on injected context size — Zep can return long summaries when the thread
# is dense. We trim before injection so the agent's prompt stays lean.
_CONTEXT_INJECT_MAX_CHARS = 1800


def _user_id_for(run_id: str, persona_id: str) -> str:
    """Stable, sluggified Zep user id for a (run, persona) pair."""
    return _slug(f"pf-sim-{run_id}-{persona_id}")


def _thread_id_for(run_id: str, persona_id: str) -> str:
    return _slug(f"pf-thread-{run_id}-{persona_id}")


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]")


def _slug(s: str) -> str:
    """Zep ids accept alphanumerics + a few separators. Replace anything else
    with `-` so timestamps with colons / persona ids with spaces never break.
    """
    return _SLUG_RE.sub("-", s)[:128]


class ZepSimMemory:
    """Thin async wrapper around Zep threads, scoped per simulation run.

    Instances are created per-run by `for_run(run_id, panel)`; the heavy
    setup (creating users + threads in Zep) happens once at construction.
    Subsequent `record_round` / `get_persona_context` calls are cheap.
    """

    def __init__(self, client: AsyncZep, run_id: str) -> None:
        self._client = client
        self._run_id = run_id
        # persona_id -> message count we've pushed so far (for the per-persona cap).
        self._counts: dict[str, int] = {}
        # persona_id -> True once we've created the user + thread.
        self._ready: dict[str, bool] = {}

    # --- bootstrap ------------------------------------------------------

    async def setup(self, panel: list[Persona]) -> None:
        """Create the Zep user + thread for every persona. Idempotent — `ApiError`
        with status 400/409 (already exists) is treated as success."""
        await asyncio.gather(
            *(self._ensure_persona(p) for p in panel),
            return_exceptions=True,
        )

    async def _ensure_persona(self, persona: Persona) -> None:
        if self._ready.get(persona.id):
            return
        user_id = _user_id_for(self._run_id, persona.id)
        thread_id = _thread_id_for(self._run_id, persona.id)
        try:
            try:
                await self._client.user.add(
                    user_id=user_id,
                    first_name=persona.name,
                    metadata={
                        "run_id": self._run_id,
                        "persona_id": persona.id,
                        "segment": persona.segment_name,
                    },
                )
            except ApiError as e:
                if e.status_code not in (400, 409):
                    raise
            try:
                await self._client.thread.create(
                    thread_id=thread_id, user_id=user_id,
                )
            except ApiError as e:
                if e.status_code not in (400, 409):
                    raise
            self._ready[persona.id] = True
            self._counts.setdefault(persona.id, 0)
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.warning(
                "zep-sim: failed to set up persona %s: %s", persona.id, e,
            )

    # --- write path -----------------------------------------------------

    async def record_round(
        self,
        round_n: int,
        new_posts: Iterable[ForumPost],
        new_comments: Iterable[ForumComment],
        persona_by_id: dict[str, Persona],
    ) -> None:
        """Push the round's new posts/comments into each persona's thread.

        Groups by persona so one persona = one batched `add_messages` call,
        keeping API round-trips low. Posts and comments by other personas
        that this persona reacted to are *not* injected here — Camel's
        memory already has the env_prompt for the next step. Zep's job is
        to remember what *this* persona said and decided.
        """
        by_persona: dict[str, list[Message]] = {}

        for post in new_posts:
            persona = persona_by_id.get(post.persona_id)
            if persona is None or not self._ready.get(persona.id):
                continue
            by_persona.setdefault(persona.id, []).append(
                Message(
                    role="assistant",
                    name=persona.id,
                    content=(
                        f"[round {round_n} post] {post.content}"
                    ),
                )
            )

        for cmt in new_comments:
            persona = persona_by_id.get(cmt.persona_id)
            if persona is None or not self._ready.get(persona.id):
                continue
            by_persona.setdefault(persona.id, []).append(
                Message(
                    role="assistant",
                    name=persona.id,
                    content=(
                        f"[round {round_n} comment on {cmt.parent_post_id}] "
                        f"{cmt.content}"
                    ),
                )
            )

        if not by_persona:
            return

        await asyncio.gather(
            *(
                self._push(persona_id, msgs)
                for persona_id, msgs in by_persona.items()
            ),
            return_exceptions=True,
        )

    async def _push(self, persona_id: str, messages: list[Message]) -> None:
        # Apply the per-persona cap *before* sending — once we're over the
        # cap we drop silently. The newest messages are the most relevant.
        budget = _MAX_MESSAGES_PER_PERSONA - self._counts.get(persona_id, 0)
        if budget <= 0:
            return
        if len(messages) > budget:
            messages = messages[:budget]
        thread_id = _thread_id_for(self._run_id, persona_id)
        try:
            await self._client.thread.add_messages(
                thread_id=thread_id, messages=messages,
            )
            self._counts[persona_id] = (
                self._counts.get(persona_id, 0) + len(messages)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "zep-sim: add_messages failed for %s: %s", persona_id, e,
            )

    # --- read path ------------------------------------------------------

    async def get_persona_context(self, persona_id: str) -> str:
        """Return the Zep-summarised "what this persona has said + believes"
        block, ready to inject into the agent's system memory. Empty string
        on any error or when the persona has no thread yet."""
        if not self._ready.get(persona_id):
            return ""
        thread_id = _thread_id_for(self._run_id, persona_id)
        try:
            # `mode="basic"` skips the heavier reranker — sub-second latency
            # at our message volume, which is what we need to keep the
            # simulation fast.
            resp = await self._client.thread.get_user_context(
                thread_id=thread_id, mode="basic",
            )
            ctx = (resp.context or "").strip()
            if len(ctx) > _CONTEXT_INJECT_MAX_CHARS:
                ctx = ctx[:_CONTEXT_INJECT_MAX_CHARS] + "\n…[truncated]"
            return ctx
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "zep-sim: get_user_context failed for %s: %s", persona_id, e,
            )
            return ""


# --- module-level singleton-ish accessor -------------------------------

_async_client: AsyncZep | None = None
_lock = threading.Lock()


def _get_async_client() -> AsyncZep | None:
    """Lazily create the AsyncZep client. Returns None when ZEP_API_KEY is
    unset — caller should treat that as "memory disabled" and skip."""
    global _async_client
    if _async_client is not None:
        return _async_client
    with _lock:
        if _async_client is not None:
            return _async_client
        key = settings().zep_api_key.strip()
        if not key:
            return None
        _async_client = AsyncZep(api_key=key)
        return _async_client


async def for_run(run_id: str, panel: list[Persona]) -> ZepSimMemory | None:
    """Create + bootstrap a per-run memory store. Returns None if Zep is
    disabled (no API key) so callers can pattern: `if mem: await mem.foo()`.
    """
    client = _get_async_client()
    if client is None:
        return None
    mem = ZepSimMemory(client=client, run_id=run_id)
    await mem.setup(panel)
    return mem


__all__ = ["ZepSimMemory", "for_run"]

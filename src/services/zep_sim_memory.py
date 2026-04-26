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
One Zep user = the **authenticated app user** (their Supabase UUID). All
per-persona threads from every run that user kicks off live under that one
user, so Zep's dashboard shows ONE knowledge graph per real human, not 20
throwaway "pf-sim-*" rows per run. Threads are still per-`(run_id, persona_id)`
so each persona's recap is isolated at retrieval time.

Each pushed message carries `metadata = {persona_id, persona_name, segment}`
so that even when Zep's entity extractor surfaces facts across the user's
graph, "Lena said X" stays distinct from "Liam said Y" — preserves attribution
without splitting users.

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


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]")


def _slug(s: str) -> str:
    """Zep ids accept alphanumerics + a few separators. Replace anything else
    with `-` so UUIDs with hyphens / persona ids stay valid and timestamps
    with colons get normalised.
    """
    return _SLUG_RE.sub("-", s)[:128]


def _thread_id_for(user_id: str, run_id: str, persona_id: str) -> str:
    """Per-(user, run, persona) thread id. The user_id prefix makes thread ids
    unique across users sharing the same run_id namespace (won't happen in
    practice, but safer than relying on run_id uniqueness alone)."""
    # Take a short fingerprint of the user_id so the thread id stays under
    # Zep's 128-char ceiling even when run_id and persona_id are verbose.
    user_fp = _slug(user_id)[:16]
    return _slug(f"pf-sim-{user_fp}-{run_id}-{persona_id}")


class ZepSimMemory:
    """Thin async wrapper around Zep threads, scoped per (auth user, run).

    Instances are created per-run by `for_run(run_id, panel, user_id)`; the
    heavy setup (creating one thread per persona under the auth user) happens
    once at construction. Subsequent `record_round` / `get_persona_context`
    calls are cheap.
    """

    def __init__(self, client: AsyncZep, run_id: str, user_id: str) -> None:
        self._client = client
        self._run_id = run_id
        self._user_id = user_id
        # persona_id -> message count we've pushed so far (for the per-persona cap).
        self._counts: dict[str, int] = {}
        # persona_id -> True once we've created the thread.
        self._ready: dict[str, bool] = {}
        # Whether we've ensured the auth user exists in Zep this lifetime —
        # set once at setup, then skipped on every subsequent setup().
        self._user_ready: bool = False

    # --- bootstrap ------------------------------------------------------

    async def setup(self, panel: list[Persona]) -> None:
        """Create the auth user (idempotent) + one thread per persona under
        that user. ApiError 400/409 means "already exists" — treated as
        success in both cases.
        """
        await self._ensure_user()
        await asyncio.gather(
            *(self._ensure_persona(p) for p in panel),
            return_exceptions=True,
        )

    async def _ensure_user(self) -> None:
        """Defensive — the signup route + the post-run `ZepMemory._ensure_user`
        already create the user, but we can't *assume* they ran (CLI runs,
        signup flow that pre-dates the Zep wiring, key change mid-stream).
        Idempotent."""
        if self._user_ready:
            return
        try:
            try:
                await self._client.user.add(user_id=self._user_id)
            except ApiError as e:
                if e.status_code not in (400, 409):
                    raise
            self._user_ready = True
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "zep-sim: failed to ensure auth user %s: %s",
                self._user_id, e,
            )

    async def _ensure_persona(self, persona: Persona) -> None:
        if self._ready.get(persona.id):
            return
        thread_id = _thread_id_for(self._user_id, self._run_id, persona.id)
        try:
            try:
                await self._client.thread.create(
                    thread_id=thread_id, user_id=self._user_id,
                )
            except ApiError as e:
                if e.status_code not in (400, 409):
                    raise
            self._ready[persona.id] = True
            self._counts.setdefault(persona.id, 0)
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.warning(
                "zep-sim: failed to set up thread for persona %s: %s",
                persona.id, e,
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
                    name=persona.name,
                    content=f"[round {round_n} post] {post.content}",
                    # Zep's entity extractor sees this metadata and uses
                    # persona_id as a stable key — keeps "Lena said X"
                    # distinct from "Liam said X" in the user's graph.
                    metadata={
                        "persona_id": persona.id,
                        "persona_name": persona.name,
                        "segment": persona.segment_name,
                        "run_id": self._run_id,
                        "round": round_n,
                        "kind": "post",
                    },
                )
            )

        for cmt in new_comments:
            persona = persona_by_id.get(cmt.persona_id)
            if persona is None or not self._ready.get(persona.id):
                continue
            by_persona.setdefault(persona.id, []).append(
                Message(
                    role="assistant",
                    name=persona.name,
                    content=(
                        f"[round {round_n} comment on {cmt.parent_post_id}] "
                        f"{cmt.content}"
                    ),
                    metadata={
                        "persona_id": persona.id,
                        "persona_name": persona.name,
                        "segment": persona.segment_name,
                        "run_id": self._run_id,
                        "round": round_n,
                        "kind": "comment",
                        "parent_post_id": cmt.parent_post_id,
                    },
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
        thread_id = _thread_id_for(self._user_id, self._run_id, persona_id)
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
        on any error or when the persona has no thread yet.

        Note: `get_user_context` returns from the WHOLE user's graph (now a
        single graph for the auth user). Per-thread scoping happens implicitly
        because Zep ranks facts by relevance to the thread's recent messages,
        which are this persona's own posts. Cross-persona contamination (a
        Lena fact surfacing in Liam's recap) stays low in practice and is
        actually realistic for a panel — they observed each other's posts in
        OASIS.
        """
        if not self._ready.get(persona_id):
            return ""
        thread_id = _thread_id_for(self._user_id, self._run_id, persona_id)
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


async def for_run(
    run_id: str,
    panel: list[Persona],
    user_id: str,
) -> ZepSimMemory | None:
    """Create + bootstrap a per-run memory store under the authenticated
    user's Zep account. Returns None if Zep is disabled (no API key) or if
    `user_id` is empty / "anon" (CLI-mode runs we don't want polluting the
    real user namespace).

    `user_id` is the Supabase auth UUID — same identity the post-run
    `ZepMemory.ingest_run()` uses, so all simulation chatter and the post-run
    knowledge graph land under the same Zep user.
    """
    client = _get_async_client()
    if client is None:
        return None
    if not user_id or user_id == "anon":
        # Refuse to create a thread under "anon" — that's the CLI fallback,
        # not a real user, and we don't want to pollute Zep's user list.
        logger.info("zep-sim: skipping memory setup (no auth user)")
        return None
    mem = ZepSimMemory(client=client, run_id=run_id, user_id=user_id)
    await mem.setup(panel)
    return mem


__all__ = ["ZepSimMemory", "for_run"]

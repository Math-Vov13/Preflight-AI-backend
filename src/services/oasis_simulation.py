"""Forum simulation phase — camel-ai SocialAgents in an OASIS environment.

Each panel persona becomes a stateful `SocialAgent`. Rounds are real
`env.step()` invocations against an OASIS Twitter-style platform; SQLite is
the canonical post / comment / like store during the run, queried after
every step to compute deltas, publish `forum.*` events on our SSE bus, and
build the final `ForumThread` the rest of the pipeline expects.

Validation signals (`would_pay`, `biggest_objection`, `wants_feature`,
`switch_from`, `final_verdict`) are NOT extracted from post content — at
the end of the simulation we run a final `agent.perform_interview(...)`
pass with a strict JSON schema and stamp each persona's *latest* post.
This keeps the downstream `ValidationAgent` aggregator happy without
brittle freeform-post parsing.

Key design choices:

* Each `Persona` → one `SocialAgent`. The brief + ontology digest are
  folded into the agent's system prompt (via
  `profile.other_info.user_profile`) so agents react authentically as
  themselves to the actual product, no "founder seed post" detour.
* Twitter recsys (lighter system-prompt template than Reddit's).
* `available_actions` restricted to CREATE_POST / CREATE_COMMENT /
  LIKE_POST / DO_NOTHING — letting OASIS pick from 30 actions
  (FOLLOW, JOIN_GROUP, PURCHASE_PRODUCT…) just creates noise.
* `model_config_dict["max_tokens"] = 16000` — Camel's
  `BaseModelBackend.token_limit` reads this and uses it as the *total
  context window* for `ScoreBasedContextCreator`, not just the output
  cap. Setting it too low shreds the system prompt before the agent
  ever sees the brief.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

from sim_config import settings
from events import publish
from schemas.ontology import Ontology
from schemas.persona import Persona
from schemas.scenario import (
    ForumComment,
    ForumLike,
    ForumPost,
    ForumThread,
    Sentiment,
    Stance,
)

logger = logging.getLogger(__name__)


# How many concurrent LLM calls OASIS will fan out per env.step. Capped low
# to stay polite with SiliconFlow's rate limits during a hackathon run; can
# bump for production.
_OASIS_SEMAPHORE = 6

# Sentiment we slap on OASIS posts — the framework doesn't track sentiment
# natively, and re-classifying every post would be a second LLM-pass tax.
# We stay neutral and let the Validation phase derive sentiment from the
# structured signals (would_pay / biggest_objection are the real signal).
_DEFAULT_SENTIMENT: Sentiment = "neutral"

_INTERVIEW_PROMPT = (
    "Based on your persona and the product brief above, reply with ONLY a "
    "JSON object — no prose, no markdown fence — using EXACTLY these keys:\n"
    '  "would_pay": one of "yes" | "no" | "maybe" | "at_lower_price"\n'
    '  "biggest_objection": short phrase (2-8 words) or empty string\n'
    '  "wants_feature": short feature name (2-6 words) or empty string\n'
    '  "switch_from": competitor name or empty string\n'
    '  "final_verdict": one of "would_use" | "would_not_use" | "undecided"\n'
    "Be honest and specific to your persona. No commentary outside the JSON."
)


class OasisSimulationRunner:
    """Forum simulation driven by camel-ai SocialAgents on an OASIS env."""

    def __init__(
        self,
        model: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model or settings().simulation_model
        self.seed = seed
        # Each run gets its own SQLite file so concurrent / sequential runs
        # don't read each other's posts.
        self._work_dir: Path | None = None

    # ------------------------------------------------------------------
    # Public surface — sync wrapper around the asyncio core.
    # ------------------------------------------------------------------

    def run_forum(
        self,
        brief: str,
        ontology: Ontology,
        panel: list[Persona],
        rounds: int = 3,
    ) -> ForumThread:
        if rounds not in (1, 2, 3):
            raise ValueError(f"rounds must be 1, 2, or 3 (got {rounds})")
        return asyncio.run(self._run_forum_async(brief, ontology, panel, rounds))

    # ------------------------------------------------------------------
    # Async core.
    # ------------------------------------------------------------------

    async def _run_forum_async(
        self,
        brief: str,
        ontology: Ontology,
        panel: list[Persona],
        rounds: int,
    ) -> ForumThread:
        # Late import — pulling oasis at module import time is heavy (sentence-
        # transformers warms up) and we'd rather pay it only when the OASIS
        # path is actually in use.
        import oasis  # noqa: PLC0415

        thread = ForumThread(brief=brief, rounds=rounds)
        publish(
            "simulation.start",
            {
                "panel_size": len(panel),
                "rounds": rounds,
                "model": self.model,
                "engine": "oasis",
            },
        )
        t0 = time.time()

        self._work_dir = Path(tempfile.mkdtemp(prefix="oasis_run_"))
        db_path = self._work_dir / "forum.db"
        logger.info("oasis run dir: %s", self._work_dir)

        try:
            model = self._build_model()
            graph, persona_by_agent_id = self._build_agent_graph(
                model, brief, ontology, panel, oasis
            )

            env = oasis.make(
                agent_graph=graph,
                platform=oasis.DefaultPlatformType.TWITTER,
                database_path=str(db_path),
                semaphore=_OASIS_SEMAPHORE,
            )
            await env.reset()

            # All the panel agents' SocialAgent instances, in panel order.
            agents_in_order = [
                _resolve_agent(graph, agent_id)
                for agent_id, _ in persona_by_agent_id.items()
            ]

            # Round-by-round drive. After each step we read the SQLite delta
            # and publish forum.* events so the live frontend stays alive.
            seen_post_ids: set[int] = set()
            seen_comment_ids: set[int] = set()
            seen_like_keys: set[tuple[int, int]] = set()

            for r in range(1, rounds + 1):
                publish(
                    "round.start",
                    {"round": r, "kind": _round_kind(r)},
                )
                actions = {ag: oasis.LLMAction() for ag in agents_in_order}
                await env.step(actions)

                new_posts, new_comments, new_likes = self._read_db_delta(
                    db_path,
                    seen_post_ids,
                    seen_comment_ids,
                    seen_like_keys,
                )
                self._extend_thread_and_publish(
                    thread,
                    new_posts,
                    new_comments,
                    new_likes,
                    persona_by_agent_id,
                    round_n=r,
                )

                publish(
                    "round.done",
                    {
                        "round": r,
                        "n_posts": len(new_posts),
                        "n_comments": len(new_comments),
                        "n_likes": len(new_likes),
                    },
                )

            # ---- Structured signal extraction -----------------------------
            # OASIS posts are freeform; the Validation agent expects per-post
            # signals (would_pay, biggest_objection, …). We always pull them
            # via INTERVIEW at the end of the run — regardless of round
            # count — and stamp them onto each persona's *latest* post in
            # the thread. This is the spine of the integration: without it
            # the downstream ValidationAgent has nothing to aggregate.
            signals = await self._collect_signals(
                agents_in_order, persona_by_agent_id
            )
            self._stamp_signals_on_thread(thread, signals)

            # ---- Close the env to flush platform writers. -----------------
            await env.close()

            dt = time.time() - t0
            publish(
                "simulation.done",
                {
                    "latency_s": round(dt, 2),
                    "n_posts": len(thread.posts),
                    "n_comments": len(thread.comments),
                    "n_likes": len(thread.likes),
                    "engine": "oasis",
                },
            )
            logger.info(
                "oasis forum simulation done in %.1fs (%d posts, %d comments, %d likes)",
                dt,
                len(thread.posts),
                len(thread.comments),
                len(thread.likes),
            )
            return thread
        except Exception as e:
            logger.exception("oasis simulation failed")
            publish("simulation.error", {"error": str(e), "engine": "oasis"})
            raise

    # ------------------------------------------------------------------
    # Setup helpers.
    # ------------------------------------------------------------------

    def _build_model(self):
        from camel.models import ModelFactory  # noqa: PLC0415
        from camel.types import ModelPlatformType  # noqa: PLC0415

        cfg = settings()
        # Camel's BaseModelBackend.token_limit reads `max_tokens` from
        # model_config_dict and uses it as the *total context window* for
        # ScoreBasedContextCreator — not just the generation cap. Setting it
        # too low (e.g. 350) shreds the system prompt before the agent ever
        # sees the brief. We give Camel a generous window matched to the
        # underlying SiliconFlow model's actual capacity (Qwen3-8B = 32k,
        # Qwen2.5-72B = 32k+) and accept that this also raises the output
        # cap; OASIS rarely emits more than ~300 tokens per turn anyway.
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=self.model,
            api_key=cfg.siliconflow_api_key,
            url=cfg.siliconflow_base_url,
            model_config_dict={"temperature": 0.85, "max_tokens": 16000},
        )

    def _build_agent_graph(
        self,
        model: Any,
        brief: str,
        ontology: Ontology,
        panel: list[Persona],
        oasis: Any,
    ) -> tuple[Any, dict[int, Persona]]:
        graph = oasis.AgentGraph()
        persona_by_agent_id: dict[int, Persona] = {}

        # Restrict the action surface to what we render downstream. Letting
        # OASIS pick from 30 actions just creates noise (FOLLOW, JOIN_GROUP,
        # PURCHASE_PRODUCT…); we want forum-shaped output.
        available_actions = [
            oasis.ActionType.CREATE_POST,
            oasis.ActionType.CREATE_COMMENT,
            oasis.ActionType.LIKE_POST,
            oasis.ActionType.DO_NOTHING,
        ]

        brief_block = brief.strip()
        # Light digest of the ontology so personas can reference segments /
        # competitors / hypotheses naturally without re-deriving them.
        ontology_block = _format_ontology_digest(ontology)

        for idx, persona in enumerate(panel):
            user_profile_text = _persona_to_user_profile(
                persona, brief_block, ontology_block
            )
            user_info = oasis.UserInfo(
                user_name=persona.id,
                name=persona.name,
                profile={"other_info": {"user_profile": user_profile_text}},
                recsys_type="twitter",  # lighter system-prompt template
            )
            agent = oasis.SocialAgent(
                agent_id=idx,
                user_info=user_info,
                model=model,
                available_actions=available_actions,
                interview_record=False,  # we use perform_interview() directly
            )
            graph.add_agent(agent)
            persona_by_agent_id[idx] = persona

        logger.info(
            "oasis graph: %d agents, %d edges",
            graph.get_num_nodes(),
            graph.get_num_edges(),
        )
        return graph, persona_by_agent_id

    # ------------------------------------------------------------------
    # SQLite reading + thread building.
    # ------------------------------------------------------------------

    def _read_db_delta(
        self,
        db_path: Path,
        seen_post_ids: set[int],
        seen_comment_ids: set[int],
        seen_like_keys: set[tuple[int, int]],
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Return rows added since the last delta read, mark them as seen."""
        new_posts: list[dict] = []
        new_comments: list[dict] = []
        new_likes: list[dict] = []
        if not db_path.exists():
            return new_posts, new_comments, new_likes

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            for row in conn.execute(
                "SELECT post_id, user_id, content FROM post ORDER BY post_id"
            ):
                pid = int(row["post_id"])
                if pid in seen_post_ids:
                    continue
                seen_post_ids.add(pid)
                new_posts.append(dict(row))
            try:
                for row in conn.execute(
                    "SELECT comment_id, post_id, user_id, content FROM comment "
                    "ORDER BY comment_id"
                ):
                    cid = int(row["comment_id"])
                    if cid in seen_comment_ids:
                        continue
                    seen_comment_ids.add(cid)
                    new_comments.append(dict(row))
            except sqlite3.OperationalError:
                pass
            try:
                for row in conn.execute(
                    "SELECT user_id, post_id FROM 'like'"
                ):
                    key = (int(row["user_id"]), int(row["post_id"]))
                    if key in seen_like_keys:
                        continue
                    seen_like_keys.add(key)
                    new_likes.append(dict(row))
            except sqlite3.OperationalError:
                pass
        finally:
            conn.close()
        return new_posts, new_comments, new_likes

    def _extend_thread_and_publish(
        self,
        thread: ForumThread,
        new_posts: list[dict],
        new_comments: list[dict],
        new_likes: list[dict],
        persona_by_agent_id: dict[int, Persona],
        round_n: int,
    ) -> None:
        """Convert SQLite rows into our pydantic schemas + emit events."""
        for p in new_posts:
            persona = persona_by_agent_id.get(int(p["user_id"]))
            if persona is None:
                continue  # post by an agent we don't track (shouldn't happen)
            post = ForumPost(
                id=f"post_{persona.id}_r{round_n}_oasis_{p['post_id']}",
                persona_id=persona.id,
                round=round_n,
                content=str(p["content"] or "")[:1500],
                sentiment=_DEFAULT_SENTIMENT,
                # Signals filled in by INTERVIEW pass at end of round 3.
            )
            thread.posts.append(post)
            publish(
                "forum.post",
                {
                    "id": post.id,
                    "persona_id": post.persona_id,
                    "round": post.round,
                    "content": post.content,
                    "sentiment": post.sentiment,
                    "would_pay": post.would_pay,
                    "biggest_objection": post.biggest_objection,
                    "wants_feature": post.wants_feature,
                    "switch_from": post.switch_from,
                    "final_verdict": post.final_verdict,
                },
            )

        # Comment parent post resolution: OASIS gives us numeric post ids; we
        # need to map back to our string post ids. Easiest: scan thread.posts
        # for matching `:oasis_<post_id>` suffix.
        post_id_lookup = {
            int(p.id.rsplit("_oasis_", 1)[1]): p.id
            for p in thread.posts
            if "_oasis_" in p.id
        }
        for c in new_comments:
            persona = persona_by_agent_id.get(int(c["user_id"]))
            if persona is None:
                continue
            parent = post_id_lookup.get(int(c["post_id"]))
            if parent is None:
                continue  # comment on an unknown post; skip
            cmt = ForumComment(
                id=f"cmt_{persona.id}_r{round_n}_oasis_{c['comment_id']}",
                persona_id=persona.id,
                round=round_n,
                parent_post_id=parent,
                content=str(c["content"] or "")[:1000],
                stance=_default_stance(),
            )
            thread.comments.append(cmt)
            publish(
                "forum.comment",
                {
                    "id": cmt.id,
                    "persona_id": cmt.persona_id,
                    "round": cmt.round,
                    "parent_post_id": cmt.parent_post_id,
                    "content": cmt.content,
                    "stance": cmt.stance,
                },
            )

        for lk in new_likes:
            persona = persona_by_agent_id.get(int(lk["user_id"]))
            if persona is None:
                continue
            target_id = post_id_lookup.get(int(lk["post_id"]))
            if target_id is None:
                continue
            like = ForumLike(
                persona_id=persona.id,
                round=round_n,
                target_id=target_id,
            )
            thread.likes.append(like)
            publish(
                "forum.like",
                {
                    "persona_id": like.persona_id,
                    "round": like.round,
                    "target_id": like.target_id,
                },
            )

    # ------------------------------------------------------------------
    # Signal extraction (INTERVIEW).
    # ------------------------------------------------------------------

    async def _collect_signals(
        self,
        agents: list[Any],
        persona_by_agent_id: dict[int, Persona],
    ) -> dict[str, dict[str, Any]]:
        """Run perform_interview() on every agent in parallel, return a
        persona_id → parsed-JSON dict map. Agents that fail to return valid
        JSON contribute an empty dict — the Validation aggregator handles
        partial signal coverage gracefully."""

        async def one(agent: Any) -> tuple[str, dict[str, Any]]:
            persona = persona_by_agent_id[agent.social_agent_id]
            try:
                result = await agent.perform_interview(_INTERVIEW_PROMPT)
                content = (result or {}).get("content", "")
                parsed = _safe_parse_json(content)
                return persona.id, parsed
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "interview failed for %s: %s", persona.id, e,
                )
                return persona.id, {}

        results = await asyncio.gather(*(one(a) for a in agents))
        return dict(results)

    def _stamp_signals_on_thread(
        self,
        thread: ForumThread,
        signals: dict[str, dict[str, Any]],
    ) -> None:
        """Apply each persona's INTERVIEW JSON to their *latest* post in the
        thread. The frontend renders signal chips off these fields, and the
        Validation agent reads them for adoption / pricing / verdict
        clustering — this is the spine of the integration."""
        latest_by_persona: dict[str, ForumPost] = {}
        for post in thread.posts:
            latest_by_persona[post.persona_id] = post

        for persona_id, sig in signals.items():
            if not sig:
                continue
            target = latest_by_persona.get(persona_id)
            if target is None:
                continue
            target.would_pay = _coerce_would_pay(sig.get("would_pay"))
            target.biggest_objection = _short(sig.get("biggest_objection"))
            target.wants_feature = _short(sig.get("wants_feature"))
            target.switch_from = _short(sig.get("switch_from"))
            target.final_verdict = _coerce_final_verdict(sig.get("final_verdict"))


# ----------------------------------------------------------------------
# Module-level helpers.
# ----------------------------------------------------------------------

def _round_kind(r: int) -> str:
    return {1: "initial_reaction", 2: "peer_reactions", 3: "final_verdict"}[r]


def _resolve_agent(graph: Any, agent_id: int) -> Any:
    """AgentGraph.get_agent returns either the agent or a (id, agent) tuple
    depending on internal version. Normalise."""
    found = graph.get_agent(agent_id)
    if isinstance(found, tuple):
        return found[1]
    return found


def _format_ontology_digest(ontology: Ontology) -> str:
    parts: list[str] = []
    if ontology.segments:
        parts.append(
            "Target segments the founders are testing: "
            + ", ".join(s.name for s in ontology.segments[:6])
        )
    if ontology.competitors:
        parts.append(
            "Existing tools users mention: "
            + ", ".join(c.name for c in ontology.competitors[:6])
        )
    if ontology.features:
        parts.append(
            "Headline features in the brief: "
            + ", ".join(f.name for f in ontology.features[:8])
        )
    return "\n".join(parts)


def _persona_to_user_profile(
    persona: Persona,
    brief: str,
    ontology_digest: str,
) -> str:
    """Render the persona + brief into a single system-prompt text block.

    OASIS feeds this into the agent's system message, so the agent acts as
    *this* persona reacting to *this* brief. Keep it dense — every line is
    extra tokens at every LLM call.
    """
    stack = ", ".join(persona.current_stack[:6]) if persona.current_stack else "—"
    return (
        f"You are {persona.name}, {persona.age}, role: {persona.role} "
        f"in {persona.location}. You belong to the segment "
        f"\"{persona.segment_name}\". "
        f"Your current pain: {persona.current_pain} "
        f"You currently use: {stack}. "
        f"Your decision style: {persona.decision_making_style}. "
        f"Your willingness to pay: up to "
        f"{persona.willingness_to_pay_eur_per_month}€/month. "
        f"How you talk: {persona.voice_sample}.\n\n"
        f"# PRODUCT YOU ARE EVALUATING\n{brief}\n\n"
        f"# CONTEXT FROM THE BRIEF\n{ontology_digest}\n\n"
        f"# HOW TO ENGAGE\n"
        f"React authentically as {persona.name} — share your candid first "
        f"impression, agree or push back on others' takes, ask questions, "
        f"give the product a like only if it actually fits your situation. "
        f"Keep posts short (60-300 chars), in your own voice, no hedging "
        f"about being an AI."
    )


def _safe_parse_json(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    # Some models prepend a "thinking" preamble before the JSON. Try to
    # locate the first {…} block and parse that.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        out = json.loads(cleaned)
        return out if isinstance(out, dict) else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


# Coercers — the model occasionally drifts on enum strings. We normalize and
# fall back to "unspecified" so the schema validation still passes.
def _coerce_would_pay(v: Any):
    if v in {"yes", "no", "maybe", "at_lower_price"}:
        return v
    return "unspecified"


def _coerce_final_verdict(v: Any):
    if v in {"would_use", "would_not_use", "undecided"}:
        return v
    return "unspecified"


def _short(v: Any, max_len: int = 80) -> str:
    if not isinstance(v, str):
        return ""
    return v.strip()[:max_len]


def _default_stance() -> Stance:
    # OASIS doesn't classify comment stance; we tag everything as "elaborate"
    # which is the most neutral of the four valid options. Validation phase
    # doesn't gate on stance, so this is loss-of-information that's fine.
    return "elaborate"

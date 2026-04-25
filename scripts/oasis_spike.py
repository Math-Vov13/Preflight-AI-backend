"""Time-boxed spike: prove OASIS 0.2.5 works end-to-end with our SiliconFlow
key, in pure Python (no CSV detour), and that we can extract structured
validation signals via ActionType.INTERVIEW at the end of a sim.

Run:
    uv run python backend/scripts/oasis_spike.py

Pass criteria:
  - 3 agents built from Python UserInfo objects (no CSV)
  - 1 LLM step produces at least 1 post in the SQLite trace
  - INTERVIEW action returns parseable JSON from each agent
  - Total wall-clock < 90 s
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)-5s %(message)s")
log = logging.getLogger("oasis-spike")

# camel-ai / oasis imports happen inside main() so the module is importable
# even when the env isn't set up yet (e.g. during a "uv tree" inspection).


def build_model():
    """Build a camel ModelFactory backend pointing at SiliconFlow.

    SiliconFlow is OpenAI-compatible, so we set model_platform=OPENAI and
    override `url` + `api_key`.
    """
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType

    api_key = os.environ.get("SILICONFLOW_API_KEY", "")
    base_url = os.environ.get(
        "SILICONFLOW_BASE_URL", "https://api.siliconflow.com/v1"
    )
    if not api_key:
        raise RuntimeError("SILICONFLOW_API_KEY missing in env")

    # Stay on a small free-tier model for the spike to keep cost negligible.
    model_type = os.environ.get("OASIS_SPIKE_MODEL", "Qwen/Qwen3-8B")
    log.info("model: platform=OPENAI type=%s url=%s", model_type, base_url)
    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=model_type,
        api_key=api_key,
        url=base_url,
        # camel needs a config dict — empty is fine for default sampling.
        model_config_dict={"temperature": 0.7, "max_tokens": 400},
    )


def build_agent_graph(model):
    """Three synthetic personas wired into an AgentGraph by hand. No CSV."""
    from oasis import (
        ActionType,
        AgentGraph,
        SocialAgent,
        UserInfo,
    )

    actions = [
        ActionType.CREATE_POST,
        ActionType.CREATE_COMMENT,
        ActionType.LIKE_POST,
        ActionType.DO_NOTHING,
    ]

    profiles = [
        {
            "id": 0,
            "user_name": "marie_lyon",
            "name": "Marie",
            "description": (
                "35yo Project Manager at a digital marketing agency in Lyon, "
                "France. Remote worker, struggles to find meaningful in-person "
                "networking. Tools: Zoom, Slack, Notion. Pain: feels isolated."
            ),
            "profile": {
                "age": 35,
                "role": "Project Manager",
                "segment": "Remote Workers",
                "wtp_eur": 12,
            },
        },
        {
            "id": 1,
            "user_name": "alex_solo",
            "name": "Alex",
            "description": (
                "29yo solo founder, B2B SaaS. Travels for conferences, looks "
                "for high-signal 1-on-1 meetups. Tools: LinkedIn, Calendly. "
                "Pain: serendipitous networking is hard to engineer."
            ),
            "profile": {
                "age": 29,
                "role": "Solo Founder",
                "segment": "Solo Founders",
                "wtp_eur": 25,
            },
        },
        {
            "id": 2,
            "user_name": "lena_traveler",
            "name": "Lena",
            "description": (
                "41yo enterprise sales, frequent business traveler. Wants "
                "structured pre-conference matchmaking. Tools: LinkedIn, "
                "Whova, calendar. Pain: hotel-lobby small talk is wasted time."
            ),
            "profile": {
                "age": 41,
                "role": "Enterprise Sales",
                "segment": "Business Travellers",
                "wtp_eur": 18,
            },
        },
    ]

    graph = AgentGraph()
    for p in profiles:
        info = UserInfo(
            user_name=p["user_name"],
            name=p["name"],
            description=p["description"],
            profile=p["profile"],
            recsys_type="reddit",
        )
        agent = SocialAgent(
            agent_id=p["id"],
            user_info=info,
            model=model,
            available_actions=actions,
            interview_record=True,
        )
        graph.add_agent(agent)
    log.info("graph: %d agents, %d edges", graph.get_num_nodes(), graph.get_num_edges())
    return graph


async def amain() -> int:
    import oasis

    t0 = time.time()
    work_dir = Path(tempfile.mkdtemp(prefix="oasis_spike_"))
    db_path = work_dir / "spike.db"
    log.info("work dir: %s", work_dir)

    try:
        model = build_model()
        graph = build_agent_graph(model)

        env = oasis.make(
            agent_graph=graph,
            platform=oasis.DefaultPlatformType.REDDIT,
            database_path=str(db_path),
            semaphore=8,
        )

        # 1. Reset — initializes the platform tables.
        log.info("env.reset()…")
        await env.reset()

        # 2. Seed a brief into the simulated forum so the agents have
        #    something to react to. We use a ManualAction CREATE_POST from
        #    agent 0 (Marie) so we don't have to wait for an LLM round
        #    just to have a parent post.
        agents = list(graph.get_agents())
        a0 = agents[0][1] if isinstance(agents[0], tuple) else agents[0]

        seed_action = oasis.ManualAction(
            action_type=oasis.ActionType.CREATE_POST,
            action_args={
                "content": (
                    "PreFlight is testing a new app called ProMeetings: "
                    "structured 1-on-1 in-person meetups for remote workers, "
                    "founders, and business travellers. Pricing: 19€/month "
                    "for unlimited meetups. Would you sign up?"
                ),
            },
        )
        log.info("seeding the forum with the brief post…")
        await env.step({a0: seed_action})

        # 3. One LLM round: every agent gets a free hand.
        log.info("step 1: LLM-driven reactions for all agents…")
        all_agents = []
        for item in graph.get_agents():
            ag = item[1] if isinstance(item, tuple) else item
            all_agents.append(ag)
        actions_dict = {ag: oasis.LLMAction() for ag in all_agents}
        await env.step(actions_dict)

        # 4. Read posts + comments straight from SQLite — that's the real
        #    "did the LLM round produce content" check.
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            posts = list(conn.execute("SELECT post_id, user_id, content, created_at FROM post"))
            comments = list(conn.execute("SELECT comment_id, post_id, user_id, content FROM comment"))
            likes = list(conn.execute("SELECT user_id, post_id FROM 'like'"))
        except sqlite3.OperationalError as e:
            log.warning("table read error: %s", e)
            posts, comments, likes = [], [], []
        finally:
            conn.close()

        log.info("=== POSTS in SQLite (%d) ===", len(posts))
        for p in posts:
            log.info(" #%s u=%s | %s", p["post_id"], p["user_id"], p["content"][:140])
        log.info("=== COMMENTS (%d) ===", len(comments))
        for c in comments:
            log.info(" #%s -> post=%s u=%s | %s",
                     c["comment_id"], c["post_id"], c["user_id"], c["content"][:140])
        log.info("=== LIKES (%d) ===", len(likes))

        # 5. INTERVIEW path: bypass env.step (which returns None) and call
        #    perform_interview() directly so we get the structured signals
        #    back as Python dicts. Run them concurrently to keep wall time
        #    bounded; the env's semaphore already throttles upstream LLM
        #    calls, but our direct path needs its own gather.
        log.info("=== INTERVIEW each agent for structured signals ===")
        interview_prompt = (
            "Reply with ONLY a JSON object (no prose, no markdown fence), "
            "with EXACTLY these keys: "
            'would_pay ("yes"|"no"|"maybe"|"at_lower_price"), '
            "biggest_objection (short string or empty), "
            "wants_feature (short string or empty), "
            "switch_from (competitor name or empty), "
            'final_verdict ("would_use"|"would_not_use"|"undecided"). '
            "Base your answer on your persona and the ProMeetings brief at "
            "19€/month."
        )
        results = await asyncio.gather(
            *(ag.perform_interview(interview_prompt) for ag in all_agents),
            return_exceptions=True,
        )
        for ag, r in zip(all_agents, results):
            uname = getattr(ag.user_info, "user_name", "?")
            if isinstance(r, Exception):
                log.warning(" %s: interview failed — %s", uname, r)
                continue
            content = (r or {}).get("content", "")
            log.info(" --- %s ---", uname)
            log.info(" raw: %s", content[:400])
            # Try to parse the JSON the model returned. If the model wrapped
            # it in markdown fences we strip those defensively.
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()
            try:
                parsed = json.loads(cleaned)
                log.info(" parsed: %s", parsed)
            except Exception as e:
                log.warning(" JSON parse failed: %s", e)

        elapsed = time.time() - t0
        log.info("spike complete in %.1fs (db=%s)", elapsed, db_path)
        return 0
    finally:
        # Keep the DB around so we can inspect it manually if needed.
        log.info("artifacts: %s", work_dir)


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())

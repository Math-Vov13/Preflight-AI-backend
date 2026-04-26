"""POST /api/runs/{run_id}/chat — streaming Q&A grounded in a persisted run.

Each request is stateless: the client sends the full message history. The
backend loads the Run's artefacts (brief, ontology, panel composition,
thread highlights, validation report) and prepends a system + context
message before handing off to the chat model.

Response shape (text/event-stream):
    data: {"type": "start"}
    data: {"type": "delta", "text": "<markdown chunk>"}
    ...
    data: {"type": "done"}
    data: {"type": "error", "error": "..."}   (on failure)
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from auth import CurrentUser
from sim_config import settings
from models import preflight_db
from models.siliconflow import client
from paths import user_runs_dir
from services.zep_memory import get_memory

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=40)


SYSTEM_PROMPT = (
    "You are a product-strategy analyst helping a founder explore the results "
    "of a PreFlight validation simulation. You have the product brief, the "
    "extracted ontology, the panel composition, the simulated forum thread "
    "signals, and the full ValidationReport (adoption by segment, top "
    "objections, missing features, switching intent, pricing feedback, "
    "hypotheses verdict, red flags, MVP cuts, go/pivot/kill recommendation). "
    "You may also receive a GRAPH MEMORY block containing cross-run facts and "
    "entities extracted by the Zep knowledge graph. When graph memory is "
    "provided, prefer it over speculation and explicitly note when a pattern "
    "spans multiple runs vs. the current run only. "
    "Ground every answer in specific data — cite segment names, persona IDs, "
    "objection phrasings, numbers. Use markdown: headings, bullet lists, "
    "`code`, tables when useful. Never invent data — if the report doesn't "
    "cover something, say so and suggest how to re-run to surface it."
)

# Cross-run memory is flavor, not the primary context; keep it tight.
_GRAPH_FACT_LIMIT = 6
_GRAPH_ENTITY_LIMIT = 6


def _load_run_context(run_id: str, user_id: str) -> dict[str, Any]:
    # DB-first: pull the artefacts dict out of run_artifacts and project it
    # into the same shape the file path produces. Panel comes back wrapped
    # ({"personas": [...]}) so unwrap defensively.
    db_run = preflight_db.get_run_with_artifacts(run_id=run_id, auth_uid=user_id)
    if db_run is not None:
        arts = db_run.get("artifacts", {})
        panel_payload = (arts.get("panel") or {}).get("payload")
        if isinstance(panel_payload, dict) and "personas" in panel_payload:
            panel_list = panel_payload["personas"]
        elif isinstance(panel_payload, list):
            panel_list = panel_payload
        else:
            panel_list = []
        artefacts = {
            "brief": db_run.get("brief", ""),
            "ontology": (arts.get("ontology") or {}).get("payload") or {},
            "panel": panel_list,
            "thread": (arts.get("thread") or {}).get("payload") or {},
            "validation_report": (arts.get("validation_report") or {}).get("payload"),
        }
    else:
        art_path = user_runs_dir(user_id) / f"pre_demo_{run_id}.json"
        if not art_path.exists():
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        try:
            artefacts = json.loads(art_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Could not read run: {e}") from e

    # Compact thread digest — full thread can be 10k+ tokens; the structured
    # aggregations in ValidationReport already reduce it. We include just the
    # top-5 most-engaged posts for extra color.
    thread = artefacts.get("thread") or {}
    posts = thread.get("posts") or []
    comments = thread.get("comments") or []
    likes = thread.get("likes") or []
    post_engagement: dict[str, int] = {}
    for lk in likes:
        tid = lk.get("target_id", "")
        post_engagement[tid] = post_engagement.get(tid, 0) + 1
    for c in comments:
        pid = c.get("parent_post_id", "")
        post_engagement[pid] = post_engagement.get(pid, 0) + 1
    r1_posts = [p for p in posts if p.get("round") == 1]
    r1_posts.sort(key=lambda p: post_engagement.get(p.get("id", ""), 0), reverse=True)
    thread_highlights = [
        {
            "id": p.get("id"),
            "persona_id": p.get("persona_id"),
            "sentiment": p.get("sentiment"),
            "would_pay": p.get("would_pay"),
            "biggest_objection": p.get("biggest_objection"),
            "wants_feature": p.get("wants_feature"),
            "switch_from": p.get("switch_from"),
            "content": p.get("content", "")[:400],
        }
        for p in r1_posts[:5]
    ]

    panel = artefacts.get("panel") or []
    by_segment: dict[str, int] = {}
    for p in panel:
        seg = p.get("segment_name", "?")
        by_segment[seg] = by_segment.get(seg, 0) + 1

    return {
        "run_id": run_id,
        "brief": artefacts.get("brief", ""),
        "ontology_summary": (artefacts.get("ontology") or {}).get("analysis_summary", ""),
        "panel_composition": by_segment,
        "panel_total": len(panel),
        "n_posts": len(posts),
        "n_comments": len(comments),
        "n_likes": len(likes),
        "thread_highlights": thread_highlights,
        "validation_report": artefacts.get("validation_report"),
    }


def _build_context_message(ctx: dict[str, Any]) -> str:
    return (
        f"RUN CONTEXT — run_id={ctx['run_id']}\n\n"
        f"## Brief\n{ctx['brief']}\n\n"
        f"## Ontology summary\n{ctx['ontology_summary']}\n\n"
        f"## Panel\n{ctx['panel_total']} personas · by segment: "
        f"{json.dumps(ctx['panel_composition'], ensure_ascii=False)}\n\n"
        f"## Forum activity\n{ctx['n_posts']} posts · {ctx['n_comments']} "
        f"comments · {ctx['n_likes']} likes\n\n"
        f"## Top-engaged posts (round 1)\n"
        f"{json.dumps(ctx['thread_highlights'], indent=2, ensure_ascii=False)}\n\n"
        f"## Validation report\n"
        f"{json.dumps(ctx['validation_report'], indent=2, ensure_ascii=False)}\n"
    )


def _fetch_graph_memory(query: str, user_id: str) -> dict[str, list[dict[str, str]]]:
    """Parallel edges+nodes lookup in *this user's* Zep graph. Returns
        {"facts": [{"fact": str}], "entities": [{"name": str, "summary": str}]}

    Always returns a dict (never raises). If Zep is disabled or errors,
    both lists are empty and the chat falls back to run-local context.
    """
    memory = get_memory()
    if memory is None:
        return {"facts": [], "entities": []}

    def _edges() -> list[dict[str, Any]]:
        try:
            return memory.search(
                query, user_id=user_id, scope="edges", limit=_GRAPH_FACT_LIMIT,
            )
        except Exception:  # noqa: BLE001
            logger.exception("graph edges search failed for q=%r", query)
            return []

    def _nodes() -> list[dict[str, Any]]:
        try:
            return memory.search(
                query, user_id=user_id, scope="nodes", limit=_GRAPH_ENTITY_LIMIT,
            )
        except Exception:  # noqa: BLE001
            logger.exception("graph nodes search failed for q=%r", query)
            return []

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_edges = ex.submit(_edges)
        fut_nodes = ex.submit(_nodes)
        edges = fut_edges.result()
        nodes = fut_nodes.result()

    facts = [
        {"fact": str(e.get("fact", "")).strip()}
        for e in edges
        if e.get("fact")
    ]
    entities = [
        {
            "name": str(n.get("name", "")).strip(),
            "summary": str(n.get("summary", "")).strip(),
        }
        for n in nodes
        if n.get("name")
    ]
    return {"facts": facts, "entities": entities}


def _render_graph_block(mem: dict[str, list[dict[str, str]]]) -> str:
    """Compact text block appended to the run-context message. Empty string
    when nothing relevant surfaced — we don't want to mention the graph at
    all in that case, to avoid the model saying 'no cross-run data' when the
    user didn't ask about cross-run patterns."""
    if not mem["facts"] and not mem["entities"]:
        return ""
    parts = ["## GRAPH MEMORY (cross-run)"]
    if mem["facts"]:
        parts.append("\n### Facts")
        for f in mem["facts"]:
            parts.append(f"- {f['fact']}")
    if mem["entities"]:
        parts.append("\n### Entities")
        for e in mem["entities"]:
            summary = e["summary"][:220]
            parts.append(f"- **{e['name']}** — {summary}")
    return "\n".join(parts) + "\n"


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---- Persistence -----------------------------------------------------------
# We write `data/runs/{run_id}_chat.json` after each successful turn so the
# conversation survives a browser reload. Format:
#
#   {
#     "run_id": "20260425_182204",
#     "messages": [
#       {"role": "user", "content": "...", "ts": 1714000000.123},
#       {"role": "assistant", "content": "...", "ts": 1714000001.456,
#        "graph_context": {"facts": [...], "entities": [...]}}
#     ],
#     "updated_at": 1714000001.456
#   }
#
# Only chat turns are stored — the system prompt and context messages are
# rebuilt fresh on every call from the artefact, so persisting them would
# duplicate state and risk drift if we tweak SYSTEM_PROMPT later.

def _chat_file(run_id: str, user_id: str) -> Path:
    return user_runs_dir(user_id) / f"pre_demo_{run_id}_chat.json"


def _load_chat_history(run_id: str, user_id: str) -> list[dict[str, Any]]:
    # DB-first; None means "DB unavailable for this user", fall back to file.
    db_msgs = preflight_db.get_chat_history(run_id=run_id, auth_uid=user_id)
    if db_msgs is not None:
        return db_msgs

    p = _chat_file(run_id, user_id)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("chat: history file unreadable for %s", run_id)
        return []
    msgs = data.get("messages")
    return msgs if isinstance(msgs, list) else []


def _save_chat_history(
    run_id: str, user_id: str, messages: list[dict[str, Any]],
) -> None:
    """Persist chat history. DB write is best-effort; file write is the
    backup so a refresh works even if the DB is down. Atomic-ish file write:
    tmp + rename so a crash mid-write can't truncate the existing history.
    """
    preflight_db.upsert_chat_history(
        run_id=run_id, auth_uid=user_id, messages=messages,
    )

    p = _chat_file(run_id, user_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "messages": messages,
        "updated_at": time.time(),
    }
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _persist_turn(
    *,
    run_id: str,
    user_id: str,
    client_messages: list[ChatMessage],
    assistant_content: str,
    graph_mem: dict[str, list[dict[str, str]]],
) -> None:
    """Append the latest user → assistant exchange to the run's chat history.

    The client sends the full transcript on every call (`client_messages` =
    every user/assistant turn the UI knows about). We snapshot that as the
    canonical history plus the brand-new assistant turn we just streamed.
    The graph_context payload from this turn rides alongside the assistant
    message so a refresh can repaint the GraphBadge.
    """
    now = time.time()
    snapshot: list[dict[str, Any]] = []
    for m in client_messages:
        snapshot.append({"role": m.role, "content": m.content, "ts": now})
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_content,
        "ts": now,
    }
    has_graph = bool(graph_mem.get("facts")) or bool(graph_mem.get("entities"))
    if has_graph:
        payload["graph_context"] = {
            "facts": graph_mem.get("facts", []),
            "entities": graph_mem.get("entities", []),
        }
    snapshot.append(payload)
    _save_chat_history(run_id, user_id, snapshot)


@router.get("/runs/{run_id}/chat")
def get_chat_history(run_id: str, user: CurrentUser) -> dict[str, Any]:
    """Return the persisted chat turns for a run, or an empty list if none yet."""
    # 404 source-of-truth follows the same DB-first / file-fallback rule as
    # the run-context loader: if the run exists in either place, serve it.
    db_run = preflight_db.get_run_with_artifacts(run_id=run_id, auth_uid=user)
    if db_run is None:
        art_path = user_runs_dir(user) / f"pre_demo_{run_id}.json"
        if not art_path.exists():
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return {"run_id": run_id, "messages": _load_chat_history(run_id, user)}


@router.post("/runs/{run_id}/chat")
def chat_on_run(
    run_id: str, req: ChatRequest, user: CurrentUser,
) -> StreamingResponse:
    ctx = _load_run_context(run_id, user)

    # Use the latest user turn as the graph query — that's the question the
    # model is about to answer, so we want facts/entities relevant to *it*,
    # not to the whole history.
    last_user = next(
        (m.content for m in reversed(req.messages) if m.role == "user"),
        "",
    )
    graph_mem = (
        _fetch_graph_memory(last_user, user)
        if last_user
        else {"facts": [], "entities": []}
    )
    graph_block = _render_graph_block(graph_mem)

    # Per-run chat thread memory. Distinct from `_fetch_graph_memory` which
    # surfaces *cross-run* facts: this block is the rolling summary of THIS
    # conversation, so follow-ups ("expand on what you said about pricing")
    # don't require resending the full transcript every turn. Best-effort —
    # an empty string here just means we lean on `req.messages` alone.
    memory = get_memory()
    thread_id: str | None = None
    chat_context_block = ""
    if memory is not None:
        thread_id = memory.chat_thread_id(run_id, user)
        chat_context_block = memory.get_chat_context(thread_id, user)

    context_msg = _build_context_message(ctx)
    if graph_block:
        context_msg += "\n" + graph_block
    if chat_context_block:
        context_msg += (
            "\n\n## CONVERSATION MEMORY (this chat so far)\n"
            f"{chat_context_block}"
        )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context_msg},
        {
            "role": "assistant",
            "content": (
                "Got it — I have the brief, ontology, panel composition, forum "
                "highlights, and the full ValidationReport for this run. Ask me "
                "anything about why the simulated users reacted the way they did, "
                "or what would change under a different assumption."
            ),
        },
    ]
    for m in req.messages:
        messages.append({"role": m.role, "content": m.content})

    model = settings().chat_model

    def gen():
        yield _sse({"type": "start", "model": model})
        # Surface what we fed the model so the UI can show a badge. Emitted
        # even when empty so clients can reset the indicator between turns.
        yield _sse({
            "type": "graph_context",
            "n_facts": len(graph_mem["facts"]),
            "n_entities": len(graph_mem["entities"]),
            "facts": graph_mem["facts"],
            "entities": graph_mem["entities"],
        })
        # Accumulate the assistant response so we can persist the full turn at
        # end-of-stream. Persistence is best-effort — a write failure must not
        # corrupt the user's view of the conversation; we already streamed the
        # answer, the on-disk copy is for refresh-survival only.
        assistant_chunks: list[str] = []
        try:
            for delta in client().chat_stream(
                model=model,
                messages=messages,
                temperature=0.4,
                max_tokens=2500,
            ):
                assistant_chunks.append(delta)
                yield _sse({"type": "delta", "text": delta})
        except Exception as e:
            logger.exception("chat stream failed")
            yield _sse({"type": "error", "error": str(e)})
            return

        full_answer = "".join(assistant_chunks)
        try:
            _persist_turn(
                run_id=run_id,
                user_id=user,
                client_messages=req.messages,
                assistant_content=full_answer,
                graph_mem=graph_mem,
            )
        except Exception:  # noqa: BLE001
            logger.exception("chat: failed to persist turns for %s", run_id)

        # Push the latest exchange into the Zep thread so the next turn's
        # `get_chat_context` reflects it. Best-effort — the local DB +
        # JSON copy above is the source of truth for refresh-survival.
        if memory is not None and thread_id is not None and last_user:
            memory.add_chat_turn(
                thread_id=thread_id,
                user_id=user,
                user_message=last_user,
                assistant_message=full_answer,
            )

        yield _sse({"type": "done"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

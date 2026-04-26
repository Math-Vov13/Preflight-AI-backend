"""Orchestrated PreFlight pipeline.

Where `pipeline.run_full_pipeline` runs a hardcoded 5-phase ladder, this
module executes a *dynamic* sequence of steps emitted by the chat LLM
(see web-frontend `src/lib/orchestratorSchema.ts` and the route handler at
`src/app/(server)/api-client/preflight/route.tsx`). Step types:

  panel       — generate Personas. metadata: {} (size + models are server-fixed
                so the orchestrator can't tune the panel).
  simulation  — run OASIS forum. metadata: {goal: str}
  review      — run a custom-prompted analysis on a prior simulation's output.
                metadata: {system_prompt: str, target_step_id: str}
  judge       — synthesise a verdict from one or more prior steps.
                metadata: {connected_steps_id: list[str]}

The executor publishes generic events that the frontend reducer
consumes by `step_id`:

  step.start  { run_id, step_id, step_type, name }
  step.update { run_id, step_id, step_type, set?, append?, payload?, details? }
  step.done   { run_id, step_id, step_type, latency_s, summary?, payload }
  step.error  { run_id, step_id, step_type, error }
  run.start / run.done / run.error  (lifecycle, identical to run_full_pipeline)

Per-step incremental updates during simulation come for free via the
existing `forum.post` / `persona.created` / etc. publishes from the
underlying services — those reach the SSE bus too. The FE reducer
currently ignores them; if we want live forum updates, route them to a
`step.update` here. Kept simple for v1.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from sim_config import settings
from events import publish, set_run_user, start_recording
from metrics.cost import get_tracker
from models import preflight_db
from models.siliconflow import client as siliconflow_client
from schemas.ontology import Ontology
from schemas.persona import Persona
from schemas.scenario import ForumThread
from services.oasis_simulation import OasisSimulationRunner
from services.ontology_generator import OntologyGenerator
from services.persona_generator import PersonaGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — orchestrator can't tune these per the user spec.
# ---------------------------------------------------------------------------

DEFAULT_PANEL_SIZE = 10
DEFAULT_SIMULATION_ROUNDS = 2
DEFAULT_SIMULATION_SEED = 42

# Hard cap to bound runtime regardless of what the orchestrator emits.
MAX_STEPS = 12

# Step-type whitelist; mirrors the FE Zod discriminated union.
STEP_TYPES = frozenset({"panel", "simulation", "review", "judge"})


# ---------------------------------------------------------------------------
# Step result + state
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    step_id: str
    step_type: str
    name: str
    description: str
    metadata: dict[str, Any]
    status: str = "pending"  # pending | active | done | error
    payload: dict[str, Any] = field(default_factory=dict)
    summary: str | None = None
    latency_s: float = 0.0
    error: str | None = None


@dataclass
class OrchestratedRunResult:
    run_id: str
    user_id: str
    brief: str
    steps: list[StepResult]
    verdict: str | None
    rationale: str | None
    total_latency_s: float
    events_log: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class OrchestrationError(ValueError):
    """Raised on a malformed `steps[]` payload — surfaces as a 400 to the FE."""


def validate_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Light structural validation of the FE-supplied steps list.

    The chat LLM already produces this with provider schema-validation +
    Zod refinement on the FE side — we re-validate here so a malformed
    payload from a misbehaving client (or a bypassed FE) doesn't crash
    halfway through the pipeline.
    """
    if not isinstance(steps, list) or not steps:
        raise OrchestrationError("steps must be a non-empty list")
    if len(steps) > MAX_STEPS:
        raise OrchestrationError(f"too many steps (max {MAX_STEPS})")

    seen_ids: set[str] = set()
    panel_seen = False
    judge_count = 0

    for idx, raw in enumerate(steps):
        if not isinstance(raw, dict):
            raise OrchestrationError(f"steps[{idx}] must be an object")
        sid = raw.get("id")
        stype = raw.get("type")
        name = raw.get("name")
        desc = raw.get("description")
        meta = raw.get("metadata", {}) or {}
        if not isinstance(sid, str) or not sid:
            raise OrchestrationError(f"steps[{idx}].id must be a non-empty string")
        if sid in seen_ids:
            raise OrchestrationError(f"steps[{idx}].id duplicated: {sid!r}")
        seen_ids.add(sid)
        if stype not in STEP_TYPES:
            raise OrchestrationError(
                f"steps[{idx}].type must be one of {sorted(STEP_TYPES)}, got {stype!r}",
            )
        if not isinstance(name, str) or not name:
            raise OrchestrationError(f"steps[{idx}].name must be a non-empty string")
        if not isinstance(desc, str) or not desc:
            raise OrchestrationError(f"steps[{idx}].description must be a non-empty string")
        if not isinstance(meta, dict):
            raise OrchestrationError(f"steps[{idx}].metadata must be an object")

        if stype == "panel":
            panel_seen = True
        elif stype == "simulation":
            if not panel_seen:
                raise OrchestrationError(
                    f"steps[{idx}] (simulation) must come after a panel step",
                )
            if not isinstance(meta.get("goal"), str) or not meta["goal"]:
                raise OrchestrationError(
                    f"steps[{idx}].metadata.goal must be a non-empty string",
                )
        elif stype == "review":
            if not isinstance(meta.get("system_prompt"), str) or not meta["system_prompt"]:
                raise OrchestrationError(
                    f"steps[{idx}].metadata.system_prompt must be a non-empty string",
                )
            target = meta.get("target_step_id")
            if not isinstance(target, str) or target not in seen_ids or target == sid:
                raise OrchestrationError(
                    f"steps[{idx}].metadata.target_step_id must reference a prior step id",
                )
        elif stype == "judge":
            judge_count += 1
            connected = meta.get("connected_steps_id")
            if not isinstance(connected, list) or not connected:
                raise OrchestrationError(
                    f"steps[{idx}].metadata.connected_steps_id must be a non-empty list",
                )
            for ref in connected:
                if not isinstance(ref, str) or ref not in seen_ids or ref == sid:
                    raise OrchestrationError(
                        f"steps[{idx}].metadata.connected_steps_id contains an unknown id: {ref!r}",
                    )

    if not panel_seen:
        raise OrchestrationError("at least one panel step is required")
    if judge_count != 1:
        raise OrchestrationError(f"exactly one judge step required, got {judge_count}")
    if steps[-1].get("type") != "judge":
        raise OrchestrationError("the last step must be of type 'judge'")

    return steps


# ---------------------------------------------------------------------------
# Step executors
# ---------------------------------------------------------------------------


def _execute_panel(
    step: StepResult,
    brief: str,
    state: dict[str, StepResult],
    *,
    cached_ontology: list[Ontology],
) -> None:
    """Generate ontology (cached for the run) + persona panel."""
    if not cached_ontology:
        ont = OntologyGenerator().generate(brief)
        cached_ontology.append(ont)
    ontology = cached_ontology[0]
    panel = PersonaGenerator().generate_panel(ontology, total_n=DEFAULT_PANEL_SIZE)
    step.payload = {
        "personas": [p.model_dump() for p in panel],
        "ontology_summary": ontology.analysis_summary,
        "n_personas": len(panel),
    }
    step.summary = f"{len(panel)} personas across {len(ontology.segments)} segments"


def _execute_simulation(
    step: StepResult,
    brief: str,
    state: dict[str, StepResult],
    *,
    cached_ontology: list[Ontology],
) -> None:
    """Run the OASIS forum on the most-recent completed panel."""
    panel_step = _resolve_most_recent(state, "panel")
    if panel_step is None:
        raise OrchestrationError("simulation step requires a completed panel step")
    persona_dicts = panel_step.payload.get("personas", [])
    panel = [Persona.model_validate(p) for p in persona_dicts]
    if not cached_ontology:
        # Defensive — should always be populated by the panel step.
        cached_ontology.append(OntologyGenerator().generate(brief))
    ontology = cached_ontology[0]

    goal = step.metadata.get("goal", "")
    # Project the orchestrator's goal into the brief so simulated agents
    # frame their reactions around it. We don't rewrite the original
    # brief — we suffix a "focus" instruction the agents pick up.
    augmented_brief = (
        f"{brief}\n\n[Simulation focus: {goal}]" if goal else brief
    )

    thread: ForumThread = OasisSimulationRunner(
        seed=DEFAULT_SIMULATION_SEED,
    ).run_forum(augmented_brief, ontology, panel, rounds=DEFAULT_SIMULATION_ROUNDS)

    step.payload = {
        "goal": goal,
        "posts": [p.model_dump() for p in thread.posts],
        "comments": [c.model_dump() for c in thread.comments],
        "likes": [l.model_dump() for l in thread.likes],
        "personas": {
            p.id: {
                "id": p.id,
                "name": p.name,
                "role": p.role,
                "location": p.location,
                "segment": p.segment_name,
            } for p in panel
        },
        "totalRounds": DEFAULT_SIMULATION_ROUNDS,
        "activeRound": None,
    }
    step.summary = (
        f"{len(thread.posts)} posts · {len(thread.comments)} comments · "
        f"{len(thread.likes)} likes"
    )


_REVIEW_USER_TEMPLATE = """BRIEF:
{brief}

SIMULATION GOAL:
{goal}

FORUM CONTENT (panel reactions):
{forum_summary}

Apply the analysis grid defined in your system prompt and return your review."""


def _execute_review(
    step: StepResult,
    brief: str,
    state: dict[str, StepResult],
) -> None:
    """Direct LLM call with the orchestrator-supplied system_prompt against
    the forum content of the targeted simulation step."""
    target_id = step.metadata["target_step_id"]
    target = state.get(target_id)
    if target is None or target.status != "done":
        raise OrchestrationError(
            f"review target '{target_id}' has not completed; can't run review yet",
        )
    if target.step_type != "simulation":
        raise OrchestrationError(
            f"review target '{target_id}' must be a simulation step, got {target.step_type}",
        )

    forum_summary = _summarize_forum(target.payload)
    user_msg = _REVIEW_USER_TEMPLATE.format(
        brief=brief[:500],
        goal=target.payload.get("goal", ""),
        forum_summary=forum_summary,
    )
    res = siliconflow_client().chat(
        model=settings().report_model,
        messages=[
            {"role": "system", "content": step.metadata["system_prompt"]},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    step.payload = {
        "system_prompt": step.metadata["system_prompt"],
        "target_step_id": target_id,
        "result": res.text,
    }
    step.summary = f"{len(res.text)} chars review"


_JUDGE_SYSTEM_PROMPT = (
    "You synthesize a final go/pivot/kill verdict from one or more upstream "
    "analyses (reviews or simulations) of a product idea. Be decisive and "
    "ground every claim in the inputs you are shown. Output JSON only."
)


_JUDGE_USER_TEMPLATE = """BRIEF:
{brief}

UPSTREAM CONTENT (reviews and/or simulations to consume):
{connected_summary}

Output JSON ONLY with this exact schema:
{{
  "verdict": "go" | "pivot" | "kill",
  "rationale": "<2-4 sentence justification>",
  "scores": {{
    "specificity": {{ "score": 0-5, "rationale": "<one line>" }},
    "evidence_grounding": {{ "score": 0-5, "rationale": "<one line>" }},
    "actionability": {{ "score": 0-5, "rationale": "<one line>" }},
    "coverage": {{ "score": 0-5, "rationale": "<one line>" }}
  }}
}}"""


def _execute_judge(
    step: StepResult,
    brief: str,
    state: dict[str, StepResult],
) -> None:
    """Synthesise verdict from the connected steps' payloads via the judge LLM."""
    connected_ids = step.metadata["connected_steps_id"]
    chunks: list[str] = []
    for cid in connected_ids:
        target = state.get(cid)
        if target is None or target.status != "done":
            raise OrchestrationError(
                f"judge connected_step '{cid}' has not completed",
            )
        chunks.append(_summarize_step_for_judge(target))
    connected_summary = "\n\n---\n\n".join(chunks)

    user_msg = _JUDGE_USER_TEMPLATE.format(
        brief=brief[:500],
        connected_summary=connected_summary[:8000],
    )
    res = siliconflow_client().chat(
        model=settings().judge_model,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )
    parsed: dict[str, Any]
    try:
        parsed = json.loads(_strip_fence(res.text))
    except (json.JSONDecodeError, TypeError):
        parsed = {
            "verdict": "pivot",
            "rationale": "Judge LLM returned malformed JSON; defaulting to 'pivot'.",
            "scores": {},
        }

    verdict = parsed.get("verdict") if parsed.get("verdict") in {"go", "pivot", "kill"} else None
    step.payload = {
        "connected_steps_id": connected_ids,
        "verdict": verdict,
        "rationale": parsed.get("rationale"),
        "scores": parsed.get("scores", {}),
        "raw": res.text,
    }
    step.summary = f"verdict: {verdict or 'unknown'}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_most_recent(
    state: dict[str, StepResult],
    step_type: str,
) -> StepResult | None:
    """Find the most-recently-completed step of the given type. Iteration
    order = insertion order = orchestration order, so reversing gives us
    the latest one."""
    for s in reversed(list(state.values())):
        if s.step_type == step_type and s.status == "done":
            return s
    return None


def _summarize_forum(sim_payload: dict[str, Any]) -> str:
    """Compact projection of a simulation step's forum into the prompt."""
    posts = sim_payload.get("posts", [])
    comments = sim_payload.get("comments", [])
    lines: list[str] = []
    for p in posts[:30]:
        author = p.get("persona_id", "?")
        snippet = (p.get("content") or "")[:300]
        sent = p.get("sentiment", "?")
        wp = p.get("would_pay", "?")
        obj = p.get("biggest_objection") or ""
        lines.append(
            f"[POST] {author} ({sent}, would_pay={wp}): {snippet}"
            + (f"\n  objection: {obj}" if obj else ""),
        )
    for c in comments[:15]:
        author = c.get("persona_id", "?")
        snippet = (c.get("content") or "")[:200]
        stance = c.get("stance", "?")
        lines.append(f"[COMMENT/{stance}] {author}: {snippet}")
    return "\n".join(lines) if lines else "(no forum content)"


def _summarize_step_for_judge(step: StepResult) -> str:
    """Compact text projection of a step's payload for the judge prompt."""
    head = f"## {step.name} ({step.step_type}) — {step.summary or 'done'}\n"
    if step.step_type == "review":
        result = step.payload.get("result", "")
        return head + str(result)[:3000]
    if step.step_type == "simulation":
        return head + _summarize_forum(step.payload)[:3000]
    if step.step_type == "panel":
        n = len(step.payload.get("personas", []))
        return head + f"{n} personas generated."
    return head + json.dumps(step.payload)[:1500]


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_orchestrated_pipeline(
    brief: str,
    steps: list[dict[str, Any]],
    *,
    run_id: str | None = None,
    user_id: str = "anon",
) -> OrchestratedRunResult:
    """Execute an orchestrator-supplied step sequence end-to-end.

    Mirrors `run_full_pipeline`'s contract: scopes events to `user_id`,
    captures a recorder so the run transcript can be persisted, publishes
    `run.start` / `run.done` / `run.error` lifecycle events.
    """
    rid = run_id or str(uuid4())
    set_run_user(user_id)
    events_log = start_recording()

    validated = validate_steps(steps)

    state: dict[str, StepResult] = {}
    for raw in validated:
        sr = StepResult(
            step_id=raw["id"],
            step_type=raw["type"],
            name=raw["name"],
            description=raw["description"],
            metadata=raw.get("metadata", {}) or {},
        )
        state[sr.step_id] = sr

    # Cache the ontology across panel + simulation steps. The orchestrator
    # never sees ontology as a step type — it's an implementation detail
    # of panel generation that simulation also reuses.
    cached_ontology: list[Ontology] = []

    publish(
        "run.start",
        {
            "run_id": rid,
            "n_steps": len(state),
            "step_ids": [s.step_id for s in state.values()],
            "brief_preview": brief[:200],
        },
    )
    t_run = time.time()
    final_verdict: str | None = None
    final_rationale: str | None = None

    try:
        for step in state.values():
            publish(
                "step.start",
                {
                    "run_id": rid,
                    "step_id": step.step_id,
                    "step_type": step.step_type,
                    "name": step.name,
                },
            )
            step.status = "active"
            t_step = time.time()
            try:
                if step.step_type == "panel":
                    _execute_panel(step, brief, state, cached_ontology=cached_ontology)
                elif step.step_type == "simulation":
                    _execute_simulation(step, brief, state, cached_ontology=cached_ontology)
                elif step.step_type == "review":
                    _execute_review(step, brief, state)
                elif step.step_type == "judge":
                    _execute_judge(step, brief, state)
                    final_verdict = step.payload.get("verdict")
                    final_rationale = step.payload.get("rationale")
                else:  # already validated, defensive only
                    raise OrchestrationError(f"unknown step type {step.step_type!r}")

                step.status = "done"
                step.latency_s = round(time.time() - t_step, 2)
                publish(
                    "step.done",
                    {
                        "run_id": rid,
                        "step_id": step.step_id,
                        "step_type": step.step_type,
                        "latency_s": step.latency_s,
                        "summary": step.summary,
                        "payload": step.payload,
                    },
                )
            except Exception as exc:
                step.status = "error"
                step.error = str(exc)
                step.latency_s = round(time.time() - t_step, 2)
                logger.exception("step %s (%s) failed", step.step_id, step.step_type)
                publish(
                    "step.error",
                    {
                        "run_id": rid,
                        "step_id": step.step_id,
                        "step_type": step.step_type,
                        "error": str(exc),
                    },
                )
                # Abort the rest of the run — downstream steps depend on
                # this one's payload.
                raise

        total = round(time.time() - t_run, 2)
        tracker = get_tracker()
        cost_summary = tracker.summary() if tracker is not None else {}
        publish(
            "run.done",
            {
                "run_id": rid,
                "total_latency_s": total,
                "cost_usd": (cost_summary or {}).get("total_usd"),
                "calls": (cost_summary or {}).get("calls"),
                "go_no_go": final_verdict,
                "rationale": final_rationale,
            },
        )

        return OrchestratedRunResult(
            run_id=rid,
            user_id=user_id,
            brief=brief,
            steps=list(state.values()),
            verdict=final_verdict,
            rationale=final_rationale,
            total_latency_s=total,
            events_log=list(events_log),
        )

    except Exception as exc:
        err_msg = str(exc)
        publish("run.error", {"run_id": rid, "error": err_msg})
        # Best-effort terminal stamp on the DB row so the FE sidebar
        # isn't stuck on "running".
        try:
            preflight_db.update_run_terminal(
                run_id=rid, status="error", error_message=err_msg,
            )
        except Exception:  # noqa: BLE001
            pass
        raise


def persist_orchestrated_run(result: OrchestratedRunResult) -> None:
    """Best-effort persistence of a finished orchestrated run.

    Updates the `runs` row with verdict + cost + wall time. Per-step
    artefacts are NOT written to `run_artifacts` — that table's `kind`
    enum doesn't accept `step:<id>` keys yet (FE wraps the failure in a
    try/catch). Once the enum migration ships, swap the no-op below for
    `preflight_db.upsert_artefact(rid, kind=f"step:{step.step_id}", payload=...)`.
    """
    rid = result.run_id
    tracker = get_tracker()
    cost_summary = tracker.summary() if tracker is not None else {}
    try:
        preflight_db.update_run_terminal(
            run_id=rid,
            status="done",
            verdict=result.verdict,
            cost_usd=(cost_summary or {}).get("total_usd"),
            wall_s=result.total_latency_s,
            rationale=result.rationale,
        )
    except Exception:  # noqa: BLE001
        logger.exception("persist_orchestrated_run: terminal stamp failed")

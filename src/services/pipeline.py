"""End-to-end PreFlight pipeline: Brief → Ontology → Panel → Simulation → Validation → Judge.

One function, one orchestration path, used by both the API (POST /api/runs/new)
and the CLI driver (scripts/run_pre_demo.py). Publishes `run.start` at entry
and `run.done` / `run.error` at exit so the live dashboard can draw a full
timeline without duplicating logic.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from events import publish
from metrics.cost import get_tracker
from metrics.judge import judge_report
from metrics.parquet_sink import write_call_records
from paths import user_runs_dir
from schemas.ontology import Ontology
from schemas.persona import Persona
from schemas.scenario import ForumThread
from schemas.validation_report import ValidationReport
from services.oasis_simulation import OasisSimulationRunner
from services.ontology_generator import OntologyGenerator
from services.persona_generator import PersonaGenerator
from services.validation_agent import ValidationAgent

logger = logging.getLogger(__name__)


@dataclass
class PhaseLatencies:
    ontology: float = 0.0
    panel: float = 0.0
    simulation: float = 0.0
    validation: float = 0.0
    judge: float = 0.0


@dataclass
class RunResult:
    run_id: str
    brief: str
    panel_size: int
    rounds: int
    ontology: Ontology
    panel: list[Persona]
    thread: ForumThread
    validation_report: ValidationReport
    judge_scores: dict[str, Any]
    phase_latencies: PhaseLatencies
    total_latency_s: float
    events_published: list[str] = field(default_factory=list)
    # Authenticated user that owns this run. Defaults to "anon" so CLI
    # scripts (test_simulation.py, run_pre_demo.py) keep working without
    # threading auth through every callsite.
    user_id: str = "anon"


def run_full_pipeline(
    brief: str,
    *,
    panel_size: int = 20,
    rounds: int = 3,
    run_id: str | None = None,
    simulation_seed: int | None = 42,
    user_id: str = "anon",
) -> RunResult:
    rid = run_id or time.strftime("%Y%m%d_%H%M%S")
    lat = PhaseLatencies()

    publish(
        "run.start",
        {"run_id": rid, "panel_size": panel_size, "rounds": rounds,
         "brief_preview": brief[:200]},
    )
    t0 = time.time()

    # 1. Ontology
    t_a = time.time()
    ontology = OntologyGenerator().generate(brief)
    lat.ontology = round(time.time() - t_a, 2)

    # 2. Panel
    t_b = time.time()
    panel = PersonaGenerator().generate_panel(ontology, total_n=panel_size)
    lat.panel = round(time.time() - t_b, 2)

    # 3. Simulation — camel-ai SocialAgents driving an OASIS environment.
    # Each panel persona becomes a stateful agent; rounds are real
    # `env.step()` invocations against a Twitter-style platform. Validation
    # signals (would_pay, biggest_objection, …) are extracted via a final
    # INTERVIEW pass and stamped onto each persona's latest post.
    t_c = time.time()
    thread = OasisSimulationRunner(seed=simulation_seed).run_forum(
        brief, ontology, panel, rounds=rounds
    )
    lat.simulation = round(time.time() - t_c, 2)

    # 4. Validation
    t_d = time.time()
    report = ValidationAgent().produce(brief, ontology, panel, thread)
    lat.validation = round(time.time() - t_d, 2)

    # 5. Judge (scores the report itself, independent quality check)
    t_e = time.time()
    publish("judge.start", {"run_id": rid})
    try:
        judge_scores = judge_report(brief, report.model_dump())
    except Exception as e:
        logger.warning("judge call failed: %s", e)
        publish("judge.error", {"run_id": rid, "error": str(e)})
        judge_scores = {}
    lat.judge = round(time.time() - t_e, 2)
    if judge_scores:
        publish(
            "judge.done",
            {
                "run_id": rid,
                "latency_s": lat.judge,
                "scores": judge_scores,
            },
        )

    total = round(time.time() - t0, 2)
    tracker = get_tracker()
    cost_summary = tracker.summary() if tracker is not None else {}
    publish(
        "run.done",
        {
            "run_id": rid,
            "total_latency_s": total,
            "phase_latencies": lat.__dict__,
            "cost_usd": (cost_summary or {}).get("total_usd"),
            "calls": (cost_summary or {}).get("calls"),
            "go_no_go": report.go_no_go_recommendation,
        },
    )
    logger.info(
        "pipeline done in %.1fs — verdict=%s  cost=%s",
        total, report.go_no_go_recommendation, (cost_summary or {}).get("total_usd"),
    )
    return RunResult(
        run_id=rid,
        brief=brief,
        panel_size=panel_size,
        rounds=rounds,
        ontology=ontology,
        panel=panel,
        thread=thread,
        validation_report=report,
        judge_scores=judge_scores,
        phase_latencies=lat,
        total_latency_s=total,
        user_id=user_id,
    )


def persist_run(result: RunResult, cost_summary: dict[str, Any]) -> dict[str, Path]:
    """Dump artefacts + metrics + Parquet to data/runs/{user_id}/. Returns
    a path dict.

    Files are namespaced under the authenticated user_id so two demo
    accounts don't see each other's runs. The `pre_demo_*` filename
    prefix is preserved (legacy CLI scripts + the listing endpoint
    expect it).

    After local artefacts are written, we attempt a Zep ingestion so the
    run shows up in that user's validation graph. Zep errors are isolated
    — disk artefacts are the source of truth.
    """
    user_dir = user_runs_dir(result.user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    rid = result.run_id

    artefacts_path = user_dir / f"pre_demo_{rid}.json"
    metrics_path = user_dir / f"pre_demo_{rid}_metrics.json"
    parquet_path = user_dir / f"pre_demo_{rid}_calls.parquet"

    # Artefacts: the things a reader wants to browse
    artefacts_payload = {
        "run_id": rid,
        "brief": result.brief,
        "ontology": result.ontology.model_dump(),
        "panel": [p.model_dump() for p in result.panel],
        "thread": result.thread.model_dump(),
        "validation_report": result.validation_report.model_dump(),
    }
    artefacts_path.write_text(
        json.dumps(artefacts_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Metrics: run metadata + cost + judge + phase latencies
    metrics_payload = {
        "run": {
            "run_id": rid,
            "brief_preview": result.brief[:300],
            "panel_size": result.panel_size,
            "rounds": result.rounds,
            "n_posts": len(result.thread.posts),
            "n_comments": len(result.thread.comments),
            "n_likes": len(result.thread.likes),
            "total_latency_s": result.total_latency_s,
            "phase_latencies": result.phase_latencies.__dict__,
            "go_no_go_recommendation": result.validation_report.go_no_go_recommendation,
        },
        "cost": cost_summary,
        "judge": result.judge_scores,
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    tracker = get_tracker()
    if tracker is not None:
        write_call_records(tracker.calls, parquet_path)

    # Postgres persistence (BE-PR1) — write the same payload to runs +
    # run_artifacts in addition to the on-disk files. Files stay as a
    # backup / dev path; the DB is canonical when available. Each
    # function is graceful (returns False on no DB / no UUID), so the
    # whole block is best-effort and never raises.
    from models import preflight_db  # noqa: PLC0415

    db_ok = preflight_db.insert_run(
        run_id=rid,
        auth_uid=result.user_id,
        brief=result.brief,
        panel_size=result.panel_size,
        rounds=result.rounds,
        settings={"simulation_seed": 42},  # carries over fixed fields if any
    )
    if db_ok:
        # Wholesale upsert of every kind we track. Ordering doesn't matter —
        # the unique (run_id, kind) index makes each call idempotent.
        preflight_db.upsert_artefact(
            run_id=rid, kind="ontology", payload=result.ontology.model_dump(),
        )
        preflight_db.upsert_artefact(
            run_id=rid,
            kind="panel",
            payload={"personas": [p.model_dump() for p in result.panel]},
        )
        preflight_db.upsert_artefact(
            run_id=rid, kind="thread", payload=result.thread.model_dump(),
        )
        preflight_db.upsert_artefact(
            run_id=rid,
            kind="validation_report",
            payload=result.validation_report.model_dump(),
        )
        if result.judge_scores:
            preflight_db.upsert_artefact(
                run_id=rid, kind="judge_scores", payload=result.judge_scores,
            )
        preflight_db.update_run_terminal(
            run_id=rid,
            status="done",
            verdict=result.validation_report.go_no_go_recommendation,
            cost_usd=(cost_summary or {}).get("total_usd"),
            wall_s=result.total_latency_s,
            rationale=result.validation_report.go_no_go_rationale,
        )

    # Zep ingestion is deferred until after disk writes so a crashed/slow
    # graph upload can never cost us a run. Imported lazily to avoid paying
    # the zep-cloud import cost for CLI paths that disable memory.
    from services.zep_memory import get_memory  # noqa: PLC0415

    memory = get_memory()
    if memory is not None:
        memory.ingest_run(result)
    else:
        logger.debug("zep: memory disabled (ZEP_API_KEY unset), skipping ingestion")

    return {
        "artefacts": artefacts_path,
        "metrics": metrics_path,
        "parquet": parquet_path,
    }

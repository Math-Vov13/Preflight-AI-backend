"""Replay local run artefacts into the Zep knowledge graph.

Scans `data/runs/pre_demo_*.json`, reconstructs a minimal RunResult per run,
and calls `ZepMemory.ingest_run`. Idempotent — the local ledger at
`data/runs/.zep_ingested.json` prevents double-ingestion.

Typical use: after setting ZEP_API_KEY in `.env`, run once to seed the graph
with historical runs. All subsequent runs ingest automatically via
`persist_run()`.

    uv run python backend/scripts/backfill_zep.py              # all runs
    uv run python backend/scripts/backfill_zep.py --force      # re-ingest all
    uv run python backend/scripts/backfill_zep.py --run 20260423_042542
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paths import RUNS_DIR  # noqa: E402
from schemas.ontology import Ontology  # noqa: E402
from schemas.persona import Persona  # noqa: E402
from schemas.scenario import ForumThread  # noqa: E402
from schemas.validation_report import ValidationReport  # noqa: E402
from services.pipeline import PhaseLatencies, RunResult  # noqa: E402
from services.zep_memory import forget_run, get_memory, is_enabled  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backfill_zep")


ARTEFACT_PREFIX = "pre_demo_"
ARTEFACT_SUFFIX = ".json"
# Exclude metrics files which share the prefix.
METRICS_MARKER = "_metrics.json"


def _list_artefacts() -> list[Path]:
    paths: list[Path] = []
    for p in sorted(RUNS_DIR.glob(f"{ARTEFACT_PREFIX}*{ARTEFACT_SUFFIX}")):
        if p.name.endswith(METRICS_MARKER):
            continue
        paths.append(p)
    return paths


def _load_run(artefact_path: Path) -> RunResult:
    """Reconstruct a RunResult from the on-disk artefact.

    Fields not written to disk (judge_scores, phase_latencies, total_latency_s)
    are filled with neutral placeholders — `_build_episodes` doesn't touch
    them. `rounds` is taken from thread.rounds so round-3 filtering works.
    """
    payload = json.loads(artefact_path.read_text(encoding="utf-8"))
    thread = ForumThread.model_validate(payload["thread"])
    panel = [Persona.model_validate(p) for p in payload["panel"]]
    return RunResult(
        run_id=payload["run_id"],
        brief=payload["brief"],
        panel_size=len(panel),
        rounds=thread.rounds,
        ontology=Ontology.model_validate(payload["ontology"]),
        panel=panel,
        thread=thread,
        validation_report=ValidationReport.model_validate(payload["validation_report"]),
        judge_scores={},
        phase_latencies=PhaseLatencies(),
        total_latency_s=0.0,
    )


def _run_id_from_path(artefact_path: Path) -> str:
    stem = artefact_path.stem  # pre_demo_YYYYMMDD_HHMMSS
    return stem.removeprefix(ARTEFACT_PREFIX)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--force", action="store_true",
        help="Re-ingest runs even if the local ledger marks them as done.",
    )
    ap.add_argument(
        "--run", type=str, default=None,
        help="Ingest only the given run_id (e.g. 20260423_042542).",
    )
    args = ap.parse_args()

    if not is_enabled():
        logger.error("ZEP_API_KEY is not set — aborting. Populate .env and retry.")
        return 2

    memory = get_memory()
    assert memory is not None  # is_enabled guarantees this

    artefacts = _list_artefacts()
    if args.run:
        artefacts = [p for p in artefacts if _run_id_from_path(p) == args.run]
        if not artefacts:
            logger.error("no artefact found for run_id=%s in %s", args.run, RUNS_DIR)
            return 1

    if not artefacts:
        logger.warning("no artefacts under %s — nothing to ingest", RUNS_DIR)
        return 0

    total_episodes = 0
    failed: list[str] = []

    for path in artefacts:
        run_id = _run_id_from_path(path)
        logger.info("== %s (%s) ==", run_id, path.name)
        if args.force:
            forget_run(run_id)
        try:
            result = _load_run(path)
        except Exception as e:  # noqa: BLE001
            logger.exception("failed to parse %s", path.name)
            failed.append(f"{run_id}: parse error — {e}")
            continue

        n = memory.ingest_run(result)
        if n == 0:
            # Either skipped (already ingested) or ingestion failed — both
            # paths log their own detail. We still count the run as not-failed
            # so the summary is honest.
            continue
        total_episodes += n
        logger.info("  → %d episodes ingested", n)

    logger.info("done — %d runs processed, %d total episodes pushed",
                len(artefacts), total_episodes)
    if failed:
        logger.warning("parse failures (%d):", len(failed))
        for msg in failed:
            logger.warning("  - %s", msg)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

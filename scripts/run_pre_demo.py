"""CLI driver for the full PreFlight pipeline.

Same orchestration as POST /api/runs/new, just invoked from the shell.

    uv run python backend/scripts/run_pre_demo.py              # N=10, defaults
    uv run python backend/scripts/run_pre_demo.py --n 20       # larger panel
    uv run python backend/scripts/run_pre_demo.py --n 5 --rounds 2   # smoke run
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sim_config import settings  # noqa: E402
from metrics.cost import CostTracker, set_tracker  # noqa: E402
from services.pipeline import persist_run, run_full_pipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


BRIEF = """ProMeetings is a mobile app for structured, in-person, 1-on-1 professional
meetings — targeted at remote workers, conference attendees, business travellers,
and solo founders. Users publish a 2h-7d availability window, get matched with
another professional within 10km, and confirm the meetup via GPS check-in at a
café / restaurant / bar. The app is explicitly NOT a dating app.

Pricing: freemium. Personal tier free, limited to 3 scheduled + 1 instant meetups.
Professional tier 19€/month, 3 months free trial, unlimited meetups + pro profile
+ conference mode (temporary in-conference visibility).

Core hypotheses:
- Pro-tier conversion > 10% from trial
- D7 retention > 25% on event creators
- Event completion rate > 60% (from matched to on-site confirmation)

Non-goals v1: group events, restaurant booking, social feed."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="panel size")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    tracker = CostTracker(budget_usd=settings().budget_usd)
    set_tracker(tracker)

    t0 = time.time()
    result = run_full_pipeline(BRIEF, panel_size=args.n, rounds=args.rounds)
    dt = time.time() - t0

    cost_summary = tracker.summary()
    paths = persist_run(result, cost_summary)

    report = result.validation_report
    print()
    print("=" * 80)
    print(
        f"RUN {result.run_id} — {dt:.1f}s wall, "
        f"${cost_summary['total_usd']:.4f}, {cost_summary['calls']} calls"
    )
    print("=" * 80)
    print(f"  verdict:    {report.go_no_go_recommendation.upper()}")
    print(f"  rationale:  {report.go_no_go_rationale[:160]}")
    if result.judge_scores:
        js = result.judge_scores
        total = sum(
            (v.get("score", 0) if isinstance(v, dict) else 0) for v in js.values()
        )
        print(f"  judge:      {total}/20 total  "
              f"(specificity {js.get('specificity', {}).get('score')}, "
              f"evidence {js.get('evidence_grounding', {}).get('score')}, "
              f"actionability {js.get('actionability', {}).get('score')}, "
              f"coverage {js.get('coverage', {}).get('score')})")
    print(f"\n  artefacts:  {paths['artefacts']}")
    print(f"  metrics:    {paths['metrics']}")
    print(f"  parquet:    {paths['parquet']}")
    print(f"\n  phases:     {result.phase_latencies.__dict__}")
    print(f"  cost/phase: {cost_summary['by_phase']}")


if __name__ == "__main__":
    main()

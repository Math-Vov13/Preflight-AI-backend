"""Full pipeline smoke-test: Brief → Ontology → Panel → Simulation → ValidationReport.

    uv run python backend/scripts/test_validation.py           # N=8
    uv run python backend/scripts/test_validation.py --n 15
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sim_config import settings  # noqa: E402
from metrics.cost import CostTracker, set_tracker  # noqa: E402
from services.ontology_generator import OntologyGenerator  # noqa: E402
from services.persona_generator import PersonaGenerator  # noqa: E402
from services.oasis_simulation import OasisSimulationRunner  # noqa: E402
from services.validation_agent import ValidationAgent  # noqa: E402

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
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    tracker = CostTracker(budget_usd=settings().budget_usd)
    set_tracker(tracker)

    t0 = time.time()
    ontology = OntologyGenerator().generate(BRIEF)
    t1 = time.time()
    panel = PersonaGenerator().generate_panel(ontology, total_n=args.n)
    t2 = time.time()
    thread = OasisSimulationRunner(seed=42).run_forum(BRIEF, ontology, panel, rounds=args.rounds)
    t3 = time.time()
    report = ValidationAgent().produce(BRIEF, ontology, panel, thread)
    t4 = time.time()

    print()
    print("=" * 80)
    print(f"PIPELINE DONE in {t4 - t0:.1f}s  —  "
          f"ontology {t1 - t0:.1f}s  ·  panel {t2 - t1:.1f}s  ·  "
          f"sim {t3 - t2:.1f}s  ·  validation {t4 - t3:.1f}s")
    print("=" * 80)

    print(f"\n### GO/NO-GO: {report.go_no_go_recommendation.upper()}")
    print(f"    {report.go_no_go_rationale}")

    print("\n### BRIEF SUMMARY")
    print(f"    {report.brief_summary}")

    print(f"\n### ADOPTION BY SEGMENT ({len(report.adoption_by_segment)})")
    for a in report.adoption_by_segment:
        print(f"    [{a.adoption_score}/5] {a.segment_name}: "
              f"{a.n_supporters}+ / {a.n_detractors}- / {a.n_neutral}~")
        if a.key_quote_supporter:
            print(f"          support: \"{a.key_quote_supporter[:100]}\"")
        if a.key_quote_detractor:
            print(f"          against: \"{a.key_quote_detractor[:100]}\"")

    print(f"\n### TOP OBJECTIONS ({len(report.top_objections)})")
    for o in report.top_objections:
        print(f"    [{o.severity}] ({o.frequency}×) {o.text}")
        print(f"          e.g. \"{o.example_quote[:80]}\"")
        print(f"          response: {o.likely_response}")

    print(f"\n### MISSING FEATURES ({len(report.missing_features)})")
    for m in report.missing_features:
        print(f"    ({m.requested_by_n}×) {m.feature}  [segments: {m.segments_requesting}]")

    print(f"\n### SWITCHING INTENT ({len(report.switching_intent)})")
    for s in report.switching_intent:
        print(f"    from {s.from_competitor}: {s.n_would_switch}→ / {s.n_would_not_switch}×")

    print("\n### PRICING")
    pf = report.pricing_feedback
    print(f"    floor ~{pf.floor_eur_month}€/mo, ceiling ~{pf.ceiling_eur_month}€/mo")
    print(f"    {pf.n_would_pay_announced_price}× would pay, "
          f"{pf.n_would_not_pay_announced_price}× would not")

    print(f"\n### HYPOTHESES VERDICT ({len(report.hypotheses_verdict)})")
    for h in report.hypotheses_verdict:
        print(f"    [{h.verdict:>12s} · {h.confidence}] {h.statement}")
        print(f"          {h.evidence[:150]}")

    print(f"\n### RED FLAGS ({len(report.red_flags)})")
    for rf in report.red_flags:
        print(f"    ⚠ {rf}")

    print(f"\n### MVP CUTS RECOMMENDED ({len(report.recommended_mvp_cuts)})")
    for c in report.recommended_mvp_cuts:
        print(f"    ✂ {c}")

    cost = tracker.summary()
    print(f"\n=== COST ===  ${cost['total_usd']:.4f} total  ({cost['calls']} calls)")
    print(f"  by phase: {cost['by_phase']}")

    out_dir = Path("data/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"_debug_validation_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "brief": BRIEF[:200],
                "ontology": ontology.model_dump(),
                "panel": [p.model_dump() for p in panel],
                "thread": thread.model_dump(),
                "report": report.model_dump(),
                "timings": {
                    "ontology_s": round(t1 - t0, 2),
                    "panel_s": round(t2 - t1, 2),
                    "simulation_s": round(t3 - t2, 2),
                    "validation_s": round(t4 - t3, 2),
                    "total_s": round(t4 - t0, 2),
                },
                "cost": cost,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n(dumped to {out_path})")


if __name__ == "__main__":
    main()

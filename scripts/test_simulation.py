"""Smoke-test the full Brief → Ontology → Panel → Simulation pipeline.

    uv run python backend/scripts/test_simulation.py           # N=8 (default)
    uv run python backend/scripts/test_simulation.py --n 15    # bigger panel
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sim_config import settings  # noqa: E402
from metrics.cost import CostTracker, set_tracker  # noqa: E402
from services.ontology_generator import OntologyGenerator  # noqa: E402
from services.persona_generator import PersonaGenerator  # noqa: E402
from services.oasis_simulation import OasisSimulationRunner  # noqa: E402

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
    parser.add_argument("--n", type=int, default=8, help="panel size")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    tracker = CostTracker(budget_usd=settings().budget_usd)
    set_tracker(tracker)

    # 1. Ontology
    t0 = time.time()
    ontology = OntologyGenerator().generate(BRIEF)
    t_ont = time.time() - t0

    # 2. Panel
    t1 = time.time()
    panel = PersonaGenerator().generate_panel(ontology, total_n=args.n)
    t_per = time.time() - t1

    # 3. Simulation
    t2 = time.time()
    thread = OasisSimulationRunner(seed=42).run_forum(
        BRIEF, ontology, panel, rounds=args.rounds
    )
    t_sim = time.time() - t2

    # === Summary ===
    print()
    print("=" * 70)
    print(f"ONTOLOGY   {t_ont:>6.1f}s  — {len(ontology.segments)} segments")
    print(f"PANEL      {t_per:>6.1f}s  — {len(panel)} personas")
    print(f"SIMULATION {t_sim:>6.1f}s  — {len(thread.posts)}p / "
          f"{len(thread.comments)}c / {len(thread.likes)}❤")
    print("=" * 70)

    # Round breakdown
    for r in range(1, args.rounds + 1):
        posts_r = thread.posts_by_round(r)
        sentiments = Counter(p.sentiment for p in posts_r)
        print(f"\nROUND {r}: {len(posts_r)} posts  sentiments={dict(sentiments)}")
        if posts_r:
            sample = posts_r[: min(3, len(posts_r))]
            for p in sample:
                print(f"  [{p.id}] {p.sentiment:>10s}  {p.content[:140]}")
                signals = []
                if p.would_pay != "unspecified":
                    signals.append(f"pay={p.would_pay}")
                if p.biggest_objection:
                    signals.append(f"obj=\"{p.biggest_objection}\"")
                if p.wants_feature:
                    signals.append(f"wants=\"{p.wants_feature}\"")
                if p.switch_from:
                    signals.append(f"switch={p.switch_from}")
                if p.final_verdict != "unspecified":
                    signals.append(f"verdict={p.final_verdict}")
                if signals:
                    print(f"         {' · '.join(signals)}")

    # Structured signal breakdown across all posts
    all_posts = thread.posts
    if all_posts:
        wp = Counter(p.would_pay for p in all_posts)
        print(f"\nWOULD_PAY:   {dict(wp)}")
        fv_r3 = Counter(p.final_verdict for p in thread.posts_by_round(3))
        if fv_r3:
            print(f"FINAL_VERDICT (r3): {dict(fv_r3)}")
        objections = [p.biggest_objection for p in all_posts if p.biggest_objection]
        if objections:
            print(f"OBJECTIONS ({len(objections)}): {objections[:5]}")
        features = [p.wants_feature for p in all_posts if p.wants_feature]
        if features:
            print(f"WANTS_FEATURE ({len(features)}): {features[:5]}")
        switches = [p.switch_from for p in all_posts if p.switch_from]
        if switches:
            print(f"SWITCH_FROM ({len(switches)}): {switches[:5]}")

    # Cost
    cost = tracker.summary()
    print(f"\n=== COST ===  ${cost['total_usd']:.4f} total  ({cost['calls']} calls)")
    print(f"  by phase: {cost['by_phase']}")

    # Dump
    out_dir = Path("data/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"_debug_sim_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "brief": BRIEF[:200],
                "ontology": ontology.model_dump(),
                "panel": [p.model_dump() for p in panel],
                "thread": thread.model_dump(),
                "timings": {
                    "ontology_s": round(t_ont, 2),
                    "panel_s": round(t_per, 2),
                    "simulation_s": round(t_sim, 2),
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

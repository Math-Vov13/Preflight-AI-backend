"""Smoke-test the OntologyGenerator on a short brief. Live SiliconFlow call.

    uv run python backend/scripts/test_ontology.py
"""
from __future__ import annotations

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
    tracker = CostTracker(budget_usd=settings().budget_usd)
    set_tracker(tracker)

    gen = OntologyGenerator()
    t0 = time.time()
    ontology = gen.generate(BRIEF)
    dt = time.time() - t0

    print()
    print(f"=== ONTOLOGY produced in {dt:.1f}s ===")
    print(f"  segments ({len(ontology.segments)}):")
    for s in ontology.segments:
        print(f"    - {s.name} [pain {s.pain_level}/5] — {s.description[:80]}")
    print(f"  competitors ({len(ontology.competitors)}):")
    for c in ontology.competitors:
        print(f"    - {c.name} [{c.category}] — fails because: {c.why_it_fails_for_target[:80]}")
    print(f"  features ({len(ontology.features)}):")
    for f in ontology.features:
        print(f"    - [{f.priority_in_brief}] {f.name}")
    print(f"  jobs_to_be_done ({len(ontology.jobs_to_be_done)}):")
    for j in ontology.jobs_to_be_done:
        print(f"    - {j.job[:100]}")
    print(f"  objections ({len(ontology.objections)}):")
    for o in ontology.objections:
        print(f"    - \"{o.text[:90]}\"  ← {o.likely_segments}")
    print(f"  user_hypotheses ({len(ontology.user_hypotheses)}):")
    for h in ontology.user_hypotheses:
        print(f"    - [{h.strength}] {h.statement}")
    print()
    print("Summary:")
    print(f"  {ontology.analysis_summary}")

    cost = tracker.summary()
    print()
    print(f"=== COST === ${cost['total_usd']:.4f}, {cost['calls']} calls")

    out_path = Path("data/runs/_debug_ontology.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(ontology.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n(dumped to {out_path})")


if __name__ == "__main__":
    main()

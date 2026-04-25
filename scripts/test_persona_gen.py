"""Smoke-test the full pipeline Brief → Ontology → Panel(N personas).

    uv run python backend/scripts/test_persona_gen.py           # N=10 (default)
    uv run python backend/scripts/test_persona_gen.py --n 30    # larger panel
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
    args = parser.parse_args()

    tracker = CostTracker(budget_usd=settings().budget_usd)
    set_tracker(tracker)

    t0 = time.time()
    ontology = OntologyGenerator().generate(BRIEF)
    t_ont = time.time() - t0
    print(
        f"=== ONTOLOGY {t_ont:.1f}s — {len(ontology.segments)} segments, "
        f"{len(ontology.features)} features ==="
    )

    t1 = time.time()
    panel = PersonaGenerator().generate_panel(ontology, total_n=args.n)
    t_per = time.time() - t1

    print()
    print(f"=== PANEL {t_per:.1f}s — {len(panel)} personas ===")
    by_segment: dict[str, int] = {}
    for p in panel:
        by_segment[p.segment_name] = by_segment.get(p.segment_name, 0) + 1
    for seg, n in by_segment.items():
        print(f"  {seg}: {n}")
    print()
    print("Sample personas:")
    for p in panel[: min(5, len(panel))]:
        print(f"  [{p.id}] {p.name}, {p.age}yo {p.gender}, {p.role} ({p.location})")
        print(f"        stack: {', '.join(p.current_stack[:4])}")
        print(f"        pain:  \"{p.current_pain[:120]}\"")
        print(f"        WTP:   {p.willingness_to_pay_eur_per_month}€/mo ({p.tech_attitude})")
        print(f"        voice: {p.voice_sample[:90]}")
        print()

    cost = tracker.summary()
    print(
        f"=== COST ===  ${cost['total_usd']:.4f} total "
        f"({cost['calls']} calls: "
        f"ontology=${cost['by_phase'].get('ontology', 0):.4f}, "
        f"persona=${cost['by_phase'].get('persona', 0):.4f})"
    )

    out_dir = Path("data/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"_debug_panel_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "ontology": ontology.model_dump(),
                "panel": [p.model_dump() for p in panel],
                "timings": {"ontology_s": round(t_ont, 2), "panel_s": round(t_per, 2)},
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

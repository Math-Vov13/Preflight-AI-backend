"""Generate a Panel of synthetic Personas from an Ontology.

Personas are allocated across Segments proportional to `pain_level` (the Segment
with the most acute pain gets more personas — it's the one we care most about
validating). Each Persona is produced by a single LLM call; calls are fanned
out in a ThreadPoolExecutor.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import ValidationError

from sim_config import settings
from events import publish
from metrics.cost import get_tracker
from models.siliconflow import client
from schemas.ontology import Ontology, Segment
from schemas.persona import Persona

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You generate realistic professional personas for a user-testing panel. "
    "Each persona must feel like a real individual — plausible biography, "
    "believable tooling stack, coherent voice. They will later post and react "
    "to a product idea in a simulated forum, so their `voice_sample` is load-bearing: "
    "it must capture how this specific person talks. "
    "Avoid stereotypes and generic filler. Anchor every detail in the Segment "
    "description you are given."
)


_USER_TEMPLATE = """SEGMENT:
{segment_json}

BRIEF CONTEXT (what they'll be reacting to):
{brief_summary}

Generate ONE persona that fits this Segment. Variety targets across calls in the
same Panel: spread ages 22-55, gender mix, geography within the relevant market,
a mix of tech attitudes (some early_adopter, most mainstream, a few late_majority).

Requirements:
- `current_stack`: 3-6 tools they actually use today that intersect with this product domain
- `current_pain`: 2 sentences in their own voice (first-person "I..."), grounded in the Segment description
- `voice_sample`: ONE sentence describing their tone + vocabulary tics (e.g. "formal, uses industry jargon, hedges with 'I guess'")
- `willingness_to_pay_eur_per_month`: plausible given role + segment; 0 if they'd refuse to pay

Persona id: use exactly "{persona_id}".
Segment name: use exactly "{segment_name}".

Respond with ONLY valid JSON matching this schema:
{schema_json}"""


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def _allocate_by_pain(segments: list[Segment], total_n: int) -> dict[str, int]:
    """Distribute N personas across segments proportional to pain_level.

    Ensures every segment gets at least 1 persona; rounding errors go to the
    segment with the highest pain.
    """
    if not segments:
        return {}
    weights = {s.name: max(1, s.pain_level) for s in segments}
    total_weight = sum(weights.values())
    raw = {name: total_n * w / total_weight for name, w in weights.items()}
    allocation = {name: max(1, round(r)) for name, r in raw.items()}
    # Trim / pad to hit exactly total_n.
    diff = total_n - sum(allocation.values())
    if diff != 0:
        pivot = max(segments, key=lambda s: s.pain_level).name
        allocation[pivot] = max(1, allocation[pivot] + diff)
    return allocation


class PersonaGenerator:
    def __init__(self, model: str | None = None, parallel_workers: int = 5) -> None:
        self.model = model or settings().persona_model
        self.parallel_workers = parallel_workers

    def generate_panel(self, ontology: Ontology, total_n: int) -> list[Persona]:
        if total_n < 1:
            raise ValueError(f"total_n must be >= 1, got {total_n}")
        if not ontology.segments:
            raise ValueError("ontology has no segments to sample from")

        allocation = _allocate_by_pain(ontology.segments, total_n)
        brief_summary = ontology.analysis_summary
        segment_by_name = {s.name: s for s in ontology.segments}

        publish(
            "panel.start",
            {"target_n": total_n, "allocation": allocation, "model": self.model},
        )

        tasks: list[tuple[Segment, str, str]] = []
        counter = 0
        for segment_name, n in allocation.items():
            seg = segment_by_name[segment_name]
            for _ in range(n):
                tasks.append((seg, f"P{counter}", brief_summary))
                counter += 1

        logger.info(
            "generating panel: n=%d across %d segments", len(tasks), len(allocation)
        )
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as ex:
            personas = list(
                ex.map(lambda t: self._generate_one(*t), tasks)
            )
        personas = [p for p in personas if p is not None]
        dt = time.time() - t0

        by_segment: dict[str, int] = {}
        for p in personas:
            by_segment[p.segment_name] = by_segment.get(p.segment_name, 0) + 1
        publish(
            "panel.done",
            {
                "latency_s": round(dt, 2),
                "n_personas": len(personas),
                "n_failed": total_n - len(personas),
                "by_segment": by_segment,
            },
        )
        logger.info(
            "panel generated in %.1fs (%d/%d personas, %d failed)",
            dt,
            len(personas),
            total_n,
            total_n - len(personas),
        )
        return personas

    def _generate_one(
        self, segment: Segment, persona_id: str, brief_summary: str
    ) -> Persona | None:
        segment_json = segment.model_dump_json(indent=2)
        schema_json = json.dumps(Persona.model_json_schema(), indent=2)
        prompt = _USER_TEMPLATE.format(
            segment_json=segment_json,
            brief_summary=brief_summary,
            persona_id=persona_id,
            segment_name=segment.name,
            schema_json=schema_json,
        )

        t0 = time.time()
        try:
            result = client().chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.warning("[%s] persona call failed: %s", persona_id, e)
            publish(
                "persona.error",
                {"persona_id": persona_id, "segment": segment.name, "error": str(e)},
            )
            return None
        latency = time.time() - t0

        tracker = get_tracker()
        if tracker is not None:
            tracker.record(
                model=result.model,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                phase="persona",
                agent_type="Persona",
                latency_s=latency,
            )

        text = _strip_fence(result.text)
        try:
            persona = Persona.model_validate_json(text)
        except ValidationError as first_err:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                try:
                    persona = Persona.model_validate_json(text[start : end + 1])
                except ValidationError:
                    logger.warning("[%s] invalid JSON: %s", persona_id, first_err)
                    publish(
                        "persona.error",
                        {
                            "persona_id": persona_id,
                            "segment": segment.name,
                            "error": "invalid JSON",
                        },
                    )
                    return None
            else:
                logger.warning("[%s] no JSON object in output", persona_id)
                return None

        # The LLM sometimes ignores the id/segment_name hints — force them.
        if persona.id != persona_id:
            persona = persona.model_copy(update={"id": persona_id})
        if persona.segment_name != segment.name:
            persona = persona.model_copy(update={"segment_name": segment.name})

        publish(
            "persona.created",
            {
                "persona_id": persona.id,
                "segment": persona.segment_name,
                "name": persona.name,
                "role": persona.role,
                "location": persona.location,
                "tech_attitude": persona.tech_attitude,
                "wtp_eur": persona.willingness_to_pay_eur_per_month,
            },
        )
        return persona


def allocate_by_pain(segments: list[Segment], total_n: int) -> dict[str, int]:
    """Public re-export for tests."""
    return _allocate_by_pain(segments, total_n)

"""Extract an Ontology from a product Brief via LLM.

This is the first phase of the PreFlight pipeline. The Ontology drives Persona
generation downstream, so every entity here will surface in the simulated
forum and in the final ValidationReport.
"""
from __future__ import annotations

import json
import logging
import time

from pydantic import ValidationError

from sim_config import settings
from events import publish
from metrics.cost import get_tracker
from models.siliconflow import client
from schemas.ontology import Ontology

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a product-strategy analyst. You read product briefs and extract a "
    "structured ontology: target user segments, competitors, features, jobs-to-be-done, "
    "user objections, and testable hypotheses. Your output feeds a downstream pipeline "
    "that generates synthetic personas and simulates them testing the product — every "
    "entity you extract will become an agent, a node in a knowledge graph, or a rubric "
    "item in the final validation report. "
    "Ground every item in the brief text. Do not invent segments or competitors the "
    "brief does not imply. Prefer fewer high-quality items over many speculative ones."
)


_USER_TEMPLATE = """BRIEF:
{brief}

Extract the ontology with these targets (loose bounds — produce what the brief supports, nothing more):
- segments: 2-8 differentiated target cohorts
- competitors: 1-8 existing tools / services / DIY workarounds
- features: 3-20 capabilities listed in the brief (with must/should/could priority)
- jobs_to_be_done: 1-10 jobs the product claims to address
- objections: 2-12 concerns real users would raise (NOT bugs, concerns)
- user_hypotheses: 1-10 testable claims the team is implicitly making

Rules:
- `segments[].pain_level`: 1 = mild annoyance, 3 = real problem, 5 = burning hair
- `competitors[].category`: direct | tangential | DIY | none
- `features[].priority_in_brief`: must | should | could
- `user_hypotheses[].strength`: strong (directly stated with metric) | medium (implied) | weak (inferred)

Respond with ONLY valid JSON matching this schema (no prose, no markdown fences):
{schema_json}"""


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


class OntologyGenerator:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings().ontology_model

    def generate(self, brief: str) -> Ontology:
        schema_json = json.dumps(Ontology.model_json_schema(), indent=2)
        prompt = _USER_TEMPLATE.format(brief=brief, schema_json=schema_json)

        publish(
            "ontology.start",
            {"brief_chars": len(brief), "model": self.model},
        )
        t0 = time.time()
        result = client().chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )
        latency = time.time() - t0

        tracker = get_tracker()
        if tracker is not None:
            tracker.record(
                model=result.model,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                phase="ontology",
                agent_type="Ontology",
                latency_s=latency,
            )

        text = _strip_fence(result.text)
        try:
            ontology = Ontology.model_validate_json(text)
        except ValidationError as first_err:
            # Some models wrap the object in prose; try the first {...} block.
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                try:
                    ontology = Ontology.model_validate_json(text[start : end + 1])
                except ValidationError:
                    raise RuntimeError(
                        f"Ontology parse failed after fallback: {first_err}"
                    ) from first_err
            else:
                raise RuntimeError(f"Ontology parse failed: {first_err}") from first_err

        for s in ontology.segments:
            publish(
                "ontology.entity",
                {"kind": "Segment", "name": s.name, "pain_level": s.pain_level},
            )
        for c in ontology.competitors:
            publish(
                "ontology.entity",
                {"kind": "Competitor", "name": c.name, "category": c.category},
            )
        for f in ontology.features:
            publish(
                "ontology.entity",
                {"kind": "Feature", "name": f.name, "priority": f.priority_in_brief},
            )
        for o in ontology.objections:
            publish(
                "ontology.entity",
                {"kind": "Objection", "text": o.text[:80]},
            )

        publish(
            "ontology.done",
            {
                "latency_s": round(latency, 2),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "segments": len(ontology.segments),
                "competitors": len(ontology.competitors),
                "features": len(ontology.features),
                "objections": len(ontology.objections),
                "hypotheses": len(ontology.user_hypotheses),
            },
        )
        logger.info(
            "ontology generated in %.1fs (%d segments, %d competitors, %d features)",
            latency,
            len(ontology.segments),
            len(ontology.competitors),
            len(ontology.features),
        )
        return ontology

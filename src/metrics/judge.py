"""LLM-as-judge scoring of a PreFlight ValidationReport.

Rubric is PreFlight-specific — specificity / evidence_grounding / actionability
/ coverage — replacing ai-agent-demo's completeness / coherence / novelty /
feasibility (which measured SDLC artefacts).
"""
from __future__ import annotations

import json
import time
from typing import Any

from sim_config import settings
from metrics.cost import get_tracker
from models.siliconflow import client

JUDGE_SYSTEM = (
    "You are an impartial reviewer scoring a pre-launch ValidationReport "
    "produced by a multi-agent simulation of target users. Score each "
    "rubric item 0-5. Ground every rationale in specific report content."
)

_TEMPLATE = """BRIEF:
{brief}

VALIDATION REPORT:
{report_json}

Rubric (0-5 each):
- specificity: objections, missing_features, and quotes reference concrete details from simulated personas (names, jobs, situations) — not generic hand-waving
- evidence_grounding: claims trace back to specific posts/comments in the simulated forum; nothing is invented
- actionability: findings give the product team something concrete to do (cut this feature, address this objection, re-price to X, retest hypothesis Y)
- coverage: all ValidationRubric sections are non-trivially populated (no empty lists, no placeholder scores, no "TBD")

Respond with ONLY valid JSON of this exact shape:
{{"specificity": {{"score": 0, "rationale": "..."}},
  "evidence_grounding": {{"score": 0, "rationale": "..."}},
  "actionability": {{"score": 0, "rationale": "..."}},
  "coverage": {{"score": 0, "rationale": "..."}}}}"""


def judge_report(brief: str, report: dict[str, Any]) -> dict[str, Any]:
    prompt = _TEMPLATE.format(
        brief=brief,
        report_json=json.dumps(report, indent=2, ensure_ascii=False),
    )
    model = settings().judge_model
    t0 = time.time()
    result = client().chat(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )
    latency = time.time() - t0
    tracker = get_tracker()
    if tracker is not None:
        tracker.record(
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            phase="judge",
            agent_type="Judge",
            latency_s=latency,
        )
    return json.loads(result.text)

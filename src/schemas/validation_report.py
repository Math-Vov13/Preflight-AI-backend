"""ValidationReport — the final PreFlight artefact. 6 rubric dimensions + derivatives."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["blocker", "friction", "minor"]
Verdict = Literal["validated", "invalidated", "inconclusive"]
Confidence = Literal["high", "medium", "low"]
GoNoGo = Literal["go", "pivot", "kill"]


class SegmentAdoption(BaseModel):
    segment_name: str
    adoption_score: int = Field(ge=0, le=5, description="0 = reject outright, 5 = enthusiastic")
    n_supporters: int
    n_detractors: int
    n_neutral: int
    key_quote_supporter: str
    key_quote_detractor: str


class ReportObjection(BaseModel):
    """Aggregated objection — distinct from Ontology.Objection which is *anticipated*;
    this one is *observed* in the simulated forum."""

    text: str = Field(description="Canonical phrasing of the objection")
    frequency: int = Field(description="How many personas voiced this or close variants")
    severity: Severity
    example_quote: str
    likely_response: str = Field(description="How the team could address — 1 sentence")


class MissingFeature(BaseModel):
    feature: str
    requested_by_n: int
    segments_requesting: list[str]
    example_request: str


class SwitchingIntent(BaseModel):
    from_competitor: str
    n_would_switch: int
    n_would_not_switch: int
    switching_drivers: list[str]
    resistance_factors: list[str]


class PricingFeedback(BaseModel):
    floor_eur_month: float = Field(
        description="Median minimum paid personas would pay, 0 if not enough data"
    )
    ceiling_eur_month: float = Field(
        description="Price where majority refuses, 0 if not enough data"
    )
    n_would_pay_announced_price: int
    n_would_not_pay_announced_price: int
    example_comment: str


class HypothesisVerdict(BaseModel):
    statement: str = Field(description="Restated from Ontology.user_hypotheses")
    verdict: Verdict
    evidence: str = Field(description="Specific posts/quotes supporting the verdict")
    confidence: Confidence


class ValidationReport(BaseModel):
    brief_summary: str = Field(description="2-4 sentence recap of what was tested")
    panel_composition: dict[str, int] = Field(description="segment_name -> n_personas")

    # Core 6 rubrics
    adoption_by_segment: list[SegmentAdoption]
    top_objections: list[ReportObjection] = Field(max_length=10)
    missing_features: list[MissingFeature] = Field(max_length=10)
    switching_intent: list[SwitchingIntent] = Field(max_length=8)
    pricing_feedback: PricingFeedback
    hypotheses_verdict: list[HypothesisVerdict]

    # Derivatives
    red_flags: list[str] = Field(
        description="High-severity issues: value prop confusion, legal blockers, competitor parity"
    )
    recommended_mvp_cuts: list[str] = Field(
        description="Features the panel ignored or rejected — drop from v1"
    )
    go_no_go_recommendation: GoNoGo
    go_no_go_rationale: str = Field(description="2 sentence justification")

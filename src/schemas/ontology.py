"""Ontology extracted from a product Brief — feeds Persona generation + Simulation."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PainLevel = int  # 1-5


class Segment(BaseModel):
    name: str
    description: str
    n_estimated_in_market: str = Field(
        description='Rough size estimate in the team\'s own phrasing, e.g. "~500K remote workers in France"'
    )
    pain_level: PainLevel = Field(ge=1, le=5)
    key_characteristics: list[str] = Field(
        description="3-6 traits that make this segment identifiable"
    )


class Competitor(BaseModel):
    name: str
    category: Literal["direct", "tangential", "DIY", "none"] = Field(
        description='"direct" = same job, "tangential" = partial overlap, "DIY" = build your own, "none" = users do nothing today'
    )
    why_users_use_it: str
    why_it_fails_for_target: str


class Feature(BaseModel):
    name: str
    description: str
    priority_in_brief: Literal["must", "should", "could"]
    answers_pain: str = Field(description="Which segment pain this feature addresses")


class JobToBeDone(BaseModel):
    job: str = Field(description='Phrased as "When <situation>, I want to <motivation>, so I can <outcome>"')
    current_solution: str
    gap: str


class Objection(BaseModel):
    text: str = Field(description="In the voice of a target user")
    likely_segments: list[str] = Field(
        description="Segment names most likely to raise this objection"
    )


class UserHypothesis(BaseModel):
    statement: str
    testable_metric: str = Field(
        description='E.g. "D7 retention > 25%", "Pro-tier conversion > 10%"'
    )
    strength: Literal["strong", "medium", "weak"]


class Ontology(BaseModel):
    segments: list[Segment] = Field(max_length=8)
    competitors: list[Competitor] = Field(max_length=8)
    features: list[Feature] = Field(max_length=20)
    jobs_to_be_done: list[JobToBeDone] = Field(max_length=10)
    objections: list[Objection] = Field(max_length=12)
    user_hypotheses: list[UserHypothesis] = Field(max_length=10)
    analysis_summary: str = Field(
        description="2-4 sentence executive summary of the ontology"
    )

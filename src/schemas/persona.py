"""Synthetic user persona sampled from an Ontology Segment."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Persona(BaseModel):
    id: str = Field(description='Stable identifier within a Run, e.g. "P0"')
    segment_name: str = Field(description="Name of the Segment this persona instantiates")

    # Identity
    name: str = Field(description="Fictional first name, culturally consistent with location")
    age: int = Field(ge=18, le=75)
    gender: Literal["male", "female", "other"]
    role: str = Field(description="Job title + context, e.g. 'Sales rep at SaaS B2B startup'")
    location: str = Field(description="City, country")

    # Context
    current_stack: list[str] = Field(
        description="3-6 tools they actively use today that intersect with the product domain"
    )
    current_pain: str = Field(description="Concrete frustration, 2 sentences, in their voice")
    tech_attitude: Literal["early_adopter", "mainstream", "late_majority"]
    willingness_to_pay_eur_per_month: int = Field(ge=0, le=200)
    decision_making_style: str = Field(
        description="Short descriptor, e.g. 'asks 3 peers before buying', 'signs up on impulse then cancels'"
    )

    # Voice — critical for keeping simulation output coherent per persona
    voice_sample: str = Field(
        description="How this persona talks: tone, vocabulary, typical hedge words. 1 sentence."
    )

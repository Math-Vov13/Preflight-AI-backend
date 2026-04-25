"""Forum-reaction scenario — posts, comments, likes produced by the Personas."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Sentiment = Literal["excited", "curious", "neutral", "skeptical", "critical"]
Stance = Literal["agree", "disagree", "elaborate", "question"]
WouldPay = Literal["yes", "no", "maybe", "at_lower_price", "unspecified"]
FinalVerdict = Literal["would_use", "would_not_use", "undecided", "unspecified"]


class ForumPost(BaseModel):
    """A post produced by one Persona in one round.

    Structured signal fields (would_pay, biggest_objection, wants_feature,
    switch_from, final_verdict) feed directly into ValidationReport aggregation
    in T4b — no string-tag parsing needed. Populated by the simulation prompt.
    """

    id: str = Field(description='E.g. "post_P3_r1" — persona + round')
    persona_id: str
    round: int = Field(ge=1, le=3)
    content: str = Field(description="60-400 char post in the persona's voice")
    sentiment: Sentiment

    # Structured signals (the simulation prompt asks the model for each one explicitly)
    would_pay: WouldPay = "unspecified"
    biggest_objection: str = Field(default="", description="Short phrase, 2-8 words, or empty")
    wants_feature: str = Field(default="", description="Feature name, 2-6 words, or empty")
    switch_from: str = Field(default="", description="Competitor name or empty")
    final_verdict: FinalVerdict = Field(
        default="unspecified",
        description="Only populated on round-3 posts",
    )


class ForumComment(BaseModel):
    id: str
    persona_id: str
    round: int
    parent_post_id: str
    content: str
    stance: Stance


class ForumLike(BaseModel):
    persona_id: str
    round: int
    target_id: str = Field(description="ForumPost.id or ForumComment.id")


class ForumThread(BaseModel):
    brief: str
    rounds: int
    posts: list[ForumPost] = Field(default_factory=list)
    comments: list[ForumComment] = Field(default_factory=list)
    likes: list[ForumLike] = Field(default_factory=list)

    def posts_by_round(self, round_n: int) -> list[ForumPost]:
        return [p for p in self.posts if p.round == round_n]

    def post_by_id(self, post_id: str) -> ForumPost | None:
        for p in self.posts:
            if p.id == post_id:
                return p
        return None

    def likes_for(self, target_id: str) -> int:
        return sum(1 for lk in self.likes if lk.target_id == target_id)

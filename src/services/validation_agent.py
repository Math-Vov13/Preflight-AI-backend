"""Produce a ValidationReport from a ForumThread.

Hybrid strategy: Python aggregation for counts/enums (fast, deterministic),
LLM clustering only where semantic judgment is needed (objections, missing
features, hypothesis verdicts), and a final synthesis call for red flags,
MVP cuts, and the go/pivot/kill recommendation.

Calls:
  - 3 parallel LLM calls (cluster_model, default DeepSeek-V3):
      objections, missing_features, hypotheses
  - 1 synthesis LLM call (synthesis_model, default DeepSeek-R1)
  - Pure Python otherwise
"""
from __future__ import annotations

import contextvars
import json
import logging
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import ValidationError

from sim_config import settings
from events import publish
from metrics.cost import get_tracker
from models.siliconflow import client
from schemas.ontology import Ontology
from schemas.persona import Persona
from schemas.scenario import ForumPost, ForumThread
from schemas.validation_report import (
    HypothesisVerdict,
    MissingFeature,
    PricingFeedback,
    ReportObjection,
    SegmentAdoption,
    SwitchingIntent,
    ValidationReport,
)

logger = logging.getLogger(__name__)


# =========================================================================
# Prompts
# =========================================================================

_OBJECTIONS_SYSTEM = (
    "You cluster raw objections from a simulated user forum into canonical groups. "
    "Keep each cluster specific and grounded — do not invent objections. "
    "Respond with ONLY valid JSON."
)

_OBJECTIONS_TEMPLATE = """Raw biggest_objection phrases from {n_posts} forum posts (one per line):
{raw_objections}

Relevant context (from the post content where the objection surfaced):
{excerpts}

Cluster these into 3-8 canonical ReportObjection entries. For each cluster:
- `text`: canonical phrasing (a reader-friendly objection, 5-15 words)
- `frequency`: how many raw phrases map to this cluster
- `severity`: "blocker" | "friction" | "minor"
- `example_quote`: one verbatim raw phrase that captures the cluster
- `likely_response`: ONE sentence on how the team could address it

Respond with JSON:
{{"objections": [{{"text": "...", "frequency": 3, "severity": "blocker", "example_quote": "...", "likely_response": "..."}}, ...]}}
"""


_FEATURES_SYSTEM = (
    "You cluster raw missing-feature requests from a simulated user forum into "
    "canonical features. Do not invent requests. Respond with ONLY valid JSON."
)

_FEATURES_TEMPLATE = """Raw wants_feature phrases from {n_posts} forum posts:
{raw_features}

Post context where the requests surfaced:
{excerpts}

Cluster into 3-10 canonical MissingFeature entries. For each:
- `feature`: canonical feature name (3-8 words)
- `requested_by_n`: how many raw phrases map to this cluster
- `segments_requesting`: segment names from the personas who asked (deduplicated)
- `example_request`: one verbatim raw phrase that captures the cluster

Respond with JSON:
{{"missing_features": [{{"feature": "...", "requested_by_n": 3, "segments_requesting": ["..."], "example_request": "..."}}, ...]}}
"""


_HYPOTHESES_SYSTEM = (
    "You judge whether product hypotheses are supported by simulated user feedback. "
    "Be honest — mark inconclusive when the forum didn't actually touch the hypothesis. "
    "Respond with ONLY valid JSON."
)

_HYPOTHESES_TEMPLATE = """HYPOTHESES (from the team's brief):
{hypotheses_json}

FORUM EVIDENCE (all post content from {n_posts} personas):
{forum_content}

For each hypothesis produce a HypothesisVerdict:
- `statement`: restate the hypothesis verbatim
- `verdict`: "validated" | "invalidated" | "inconclusive"
- `evidence`: 1-2 sentences citing specific posts/behaviors
- `confidence`: "high" | "medium" | "low"

Respond with JSON:
{{"hypotheses_verdict": [{{"statement": "...", "verdict": "validated", "evidence": "...", "confidence": "high"}}, ...]}}
"""


_SYNTHESIS_SYSTEM = (
    "You synthesise the outcome of a product-validation simulation into a crisp "
    "recommendation for the team. Be decisive: red flags are things that, left "
    "unaddressed, kill adoption. MVP cuts are features the simulated panel "
    "ignored or rejected. Your go/pivot/kill recommendation is binding — ground "
    "it in evidence. Respond with ONLY valid JSON."
)

_SYNTHESIS_TEMPLATE = """BRIEF:
{brief}

ONTOLOGY SUMMARY:
{ontology_summary}

PANEL COMPOSITION:
{panel_composition}

AGGREGATED RESULTS:
{aggregated}

FORUM HIGHLIGHTS (most-engaged round-1 posts + selected round-3 verdicts):
{highlights}

Produce:
- `brief_summary`: 2-4 sentences recapping what was tested (not what was found — that's the rest)
- `red_flags`: 0-6 bullets. High-severity issues: value-prop confusion, legal blockers, pricing floor below CAC hint, unanimous detractors in a target segment, etc.
- `recommended_mvp_cuts`: 0-6 features the panel ignored or rejected — cut from v1
- `go_no_go_recommendation`: exactly one of "go" | "pivot" | "kill"
- `go_no_go_rationale`: 2 sentences grounded in the aggregated results

Respond with JSON:
{{"brief_summary": "...",
  "red_flags": ["...", "..."],
  "recommended_mvp_cuts": ["...", "..."],
  "go_no_go_recommendation": "go",
  "go_no_go_rationale": "..."}}
"""


# =========================================================================
# Helpers
# =========================================================================


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def _parse_object(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


_PRICE_RE = re.compile(r"(?<![\w])(\d{1,4}(?:[.,]\d{1,2})?)\s*(?:€|eur|euros?|/mo|/month|\$)", re.IGNORECASE)


def _extract_prices(text: str) -> list[float]:
    out: list[float] = []
    for m in _PRICE_RE.finditer(text):
        try:
            out.append(float(m.group(1).replace(",", ".")))
        except ValueError:
            continue
    return [p for p in out if 0 < p < 1000]


def _persona_by_id(panel: list[Persona]) -> dict[str, Persona]:
    return {p.id: p for p in panel}


# =========================================================================
# ValidationAgent
# =========================================================================


class ValidationAgent:
    def __init__(
        self,
        synthesis_model: str | None = None,
        cluster_model: str | None = None,
    ) -> None:
        self.synthesis_model = synthesis_model or settings().report_model
        # Cluster steps use the judge-tier model by default — faster than R1 and
        # adequate for aggregation tasks that don't need deep reasoning.
        self.cluster_model = cluster_model or settings().judge_model

    # ------------------------------------------------------------------ public

    def produce(
        self,
        brief: str,
        ontology: Ontology,
        panel: list[Persona],
        thread: ForumThread,
    ) -> ValidationReport:
        publish(
            "validation.start",
            {
                "panel_size": len(panel),
                "n_posts": len(thread.posts),
                "synthesis_model": self.synthesis_model,
                "cluster_model": self.cluster_model,
            },
        )
        t0 = time.time()

        # --- Pure-Python aggregations (fast, deterministic)
        adoption = self._compute_adoption_by_segment(panel, thread)
        switching = self._compute_switching_intent(thread, ontology)
        pricing = self._compute_pricing_feedback(thread)
        panel_composition = _panel_composition(panel)
        publish("validation.section.done", {"section": "adoption", "n": len(adoption)})
        publish("validation.section.done", {"section": "switching", "n": len(switching)})
        publish("validation.section.done", {"section": "pricing"})

        # --- LLM clustering (3 parallel calls). copy_context so the
        # cost tracker + event-user contextvars propagate into the
        # workers — otherwise these LLM calls would record to the wrong
        # tracker (or none) and publish unscoped events.
        ctx = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=3) as ex:
            fut_obj = ex.submit(ctx.run, self._cluster_objections, thread)
            fut_feat = ex.submit(ctx.run, self._cluster_missing_features, panel, thread)
            fut_hyp = ex.submit(ctx.run, self._verify_hypotheses, ontology, thread)
            objections = fut_obj.result()
            missing = fut_feat.result()
            hypotheses = fut_hyp.result()
        publish("validation.section.done", {"section": "objections", "n": len(objections)})
        publish("validation.section.done", {"section": "missing_features", "n": len(missing)})
        publish("validation.section.done", {"section": "hypotheses", "n": len(hypotheses)})

        # --- Synthesis call (reasoning-heavy)
        synthesis = self._synthesize(
            brief, ontology, panel_composition, adoption, objections, missing,
            switching, pricing, hypotheses, thread,
        )
        publish("validation.section.done", {"section": "synthesis"})

        report = ValidationReport(
            brief_summary=synthesis.get("brief_summary", ""),
            panel_composition=panel_composition,
            adoption_by_segment=adoption,
            top_objections=objections,
            missing_features=missing,
            switching_intent=switching,
            pricing_feedback=pricing,
            hypotheses_verdict=hypotheses,
            red_flags=[str(x) for x in synthesis.get("red_flags", []) if x],
            recommended_mvp_cuts=[str(x) for x in synthesis.get("recommended_mvp_cuts", []) if x],
            go_no_go_recommendation=synthesis.get("go_no_go_recommendation", "pivot"),
            go_no_go_rationale=synthesis.get("go_no_go_rationale", ""),
        )

        dt = time.time() - t0
        publish(
            "validation.done",
            {
                "latency_s": round(dt, 2),
                "go_no_go": report.go_no_go_recommendation,
                "n_objections": len(report.top_objections),
                "n_red_flags": len(report.red_flags),
            },
        )
        logger.info(
            "validation done in %.1fs — %s (%d objections, %d red_flags)",
            dt, report.go_no_go_recommendation,
            len(report.top_objections), len(report.red_flags),
        )
        return report

    # ------------------------------------------------------------------ python-only aggregations

    def _compute_adoption_by_segment(
        self, panel: list[Persona], thread: ForumThread
    ) -> list[SegmentAdoption]:
        # Group personas by segment
        by_segment: dict[str, list[Persona]] = {}
        for p in panel:
            by_segment.setdefault(p.segment_name, []).append(p)

        # Index posts by persona_id (prefer round-3 post if available, else round-1)
        posts_by_persona: dict[str, ForumPost] = {}
        for r in (3, 1):
            for post in thread.posts_by_round(r):
                posts_by_persona.setdefault(post.persona_id, post)

        out: list[SegmentAdoption] = []
        for segment_name, personas in by_segment.items():
            support = 0
            detract = 0
            neutral = 0
            supporters: list[ForumPost] = []
            detractors: list[ForumPost] = []
            for persona in personas:
                post = posts_by_persona.get(persona.id)
                if post is None:
                    neutral += 1
                    continue
                is_support = (
                    post.final_verdict == "would_use"
                    or post.sentiment in ("excited", "curious")
                )
                is_detract = (
                    post.final_verdict == "would_not_use"
                    or post.sentiment in ("skeptical", "critical")
                )
                if is_support and not is_detract:
                    support += 1
                    supporters.append(post)
                elif is_detract and not is_support:
                    detract += 1
                    detractors.append(post)
                else:
                    neutral += 1

            total = support + detract + neutral
            if total == 0:
                adoption_score = 0
            else:
                # Linear weighting: supporters push toward 5, detractors toward 0
                raw = (support * 5 + neutral * 2.5 + detract * 0) / total
                adoption_score = max(0, min(5, round(raw)))

            def _longest(posts: list[ForumPost]) -> str:
                if not posts:
                    return ""
                return max(posts, key=lambda p: len(p.content)).content[:300]

            out.append(
                SegmentAdoption(
                    segment_name=segment_name,
                    adoption_score=adoption_score,
                    n_supporters=support,
                    n_detractors=detract,
                    n_neutral=neutral,
                    key_quote_supporter=_longest(supporters),
                    key_quote_detractor=_longest(detractors),
                )
            )
        return out

    def _compute_switching_intent(
        self, thread: ForumThread, ontology: Ontology
    ) -> list[SwitchingIntent]:
        # Build a normalised competitor map: lowercase → canonical name
        canonical = {c.name.lower(): c.name for c in ontology.competitors}

        would_switch: Counter[str] = Counter()
        wouldnt_switch: Counter[str] = Counter()

        for post in thread.posts:
            if not post.switch_from:
                continue
            key = post.switch_from.strip().lower()
            name = canonical.get(key, post.switch_from.strip())
            # Treat "would_use" or positive sentiment as intent to switch; the
            # persona named a competitor they'd leave for our product.
            if post.final_verdict == "would_use" or post.sentiment in ("excited", "curious"):
                would_switch[name] += 1
            elif post.final_verdict == "would_not_use" or post.sentiment in ("skeptical", "critical"):
                wouldnt_switch[name] += 1
            else:
                # Neutral mention still counts as a switch signal but ambiguous;
                # bias toward "would_switch" since the persona named it unprompted.
                would_switch[name] += 1

        names = set(would_switch) | set(wouldnt_switch)
        out: list[SwitchingIntent] = []
        for name in sorted(names, key=lambda n: -(would_switch[n] + wouldnt_switch[n])):
            out.append(
                SwitchingIntent(
                    from_competitor=name,
                    n_would_switch=would_switch[name],
                    n_would_not_switch=wouldnt_switch[name],
                    switching_drivers=[],  # T4c could add LLM-extracted drivers
                    resistance_factors=[],
                )
            )
        return out[:8]

    def _compute_pricing_feedback(self, thread: ForumThread) -> PricingFeedback:
        n_yes = 0
        n_no = 0
        example_yes = ""
        example_no = ""
        prices_mentioned: list[float] = []
        for post in thread.posts:
            if post.would_pay == "yes":
                n_yes += 1
                if not example_yes:
                    example_yes = post.content[:200]
            elif post.would_pay in ("no", "at_lower_price"):
                n_no += 1
                if not example_no:
                    example_no = post.content[:200]
            prices_mentioned.extend(_extract_prices(post.content))

        floor = 0.0
        ceiling = 0.0
        if prices_mentioned:
            prices_mentioned.sort()
            floor = round(prices_mentioned[len(prices_mentioned) // 4], 2)  # 25th pct
            ceiling = round(prices_mentioned[(len(prices_mentioned) * 3) // 4], 2)  # 75th pct

        example = example_yes or example_no or "(no pricing comment surfaced)"
        return PricingFeedback(
            floor_eur_month=floor,
            ceiling_eur_month=ceiling,
            n_would_pay_announced_price=n_yes,
            n_would_not_pay_announced_price=n_no,
            example_comment=example,
        )

    # ------------------------------------------------------------------ LLM clustering

    def _chat_json(
        self, *, system: str, prompt: str, model: str, phase: str, max_tokens: int = 2000
    ) -> dict[str, Any] | None:
        t0 = time.time()
        try:
            result = client().chat(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.warning("validation.%s call failed: %s", phase, e)
            return None
        latency = time.time() - t0
        tracker = get_tracker()
        if tracker is not None:
            tracker.record(
                model=result.model,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                phase=f"validation.{phase}",
                agent_type="ValidationAgent",
                latency_s=latency,
            )
        data = _parse_object(_strip_fence(result.text))
        if data is None:
            logger.warning("validation.%s: unparseable output", phase)
        return data

    def _cluster_objections(self, thread: ForumThread) -> list[ReportObjection]:
        raw = [p.biggest_objection for p in thread.posts if p.biggest_objection]
        if not raw:
            return []
        excerpts = [
            f"- [{p.id}, {p.sentiment}] {p.content[:180]}"
            for p in thread.posts if p.biggest_objection
        ][:20]
        prompt = _OBJECTIONS_TEMPLATE.format(
            n_posts=len(raw),
            raw_objections="\n".join(f"- {x}" for x in raw[:30]),
            excerpts="\n".join(excerpts),
        )
        data = self._chat_json(
            system=_OBJECTIONS_SYSTEM,
            prompt=prompt,
            model=self.cluster_model,
            phase="objections",
        )
        if data is None:
            return []
        raw_items = data.get("objections") or []
        out: list[ReportObjection] = []
        for item in raw_items[:10]:
            if not isinstance(item, dict):
                continue
            try:
                out.append(ReportObjection(
                    text=str(item.get("text", ""))[:200],
                    frequency=int(item.get("frequency", 1)),
                    severity=_coerce_severity(item.get("severity")),
                    example_quote=str(item.get("example_quote", ""))[:200],
                    likely_response=str(item.get("likely_response", ""))[:300],
                ))
            except (ValidationError, ValueError, TypeError):
                continue
        return out

    def _cluster_missing_features(
        self, panel: list[Persona], thread: ForumThread
    ) -> list[MissingFeature]:
        raw_entries: list[tuple[str, str]] = []  # (feature, persona_id)
        for p in thread.posts:
            if p.wants_feature:
                raw_entries.append((p.wants_feature, p.persona_id))
        if not raw_entries:
            return []
        persona_seg = {p.id: p.segment_name for p in panel}
        excerpts = [
            f"- [{p.id}, seg={persona_seg.get(p.persona_id, '?')}] wants={p.wants_feature} :: {p.content[:140]}"
            for p in thread.posts if p.wants_feature
        ][:20]
        prompt = _FEATURES_TEMPLATE.format(
            n_posts=len(raw_entries),
            raw_features="\n".join(f"- {f} (from {pid}, segment {persona_seg.get(pid, '?')})"
                                   for f, pid in raw_entries[:30]),
            excerpts="\n".join(excerpts),
        )
        data = self._chat_json(
            system=_FEATURES_SYSTEM,
            prompt=prompt,
            model=self.cluster_model,
            phase="missing_features",
        )
        if data is None:
            return []
        raw_items = data.get("missing_features") or []
        out: list[MissingFeature] = []
        for item in raw_items[:10]:
            if not isinstance(item, dict):
                continue
            segs_raw = item.get("segments_requesting") or []
            if not isinstance(segs_raw, list):
                segs_raw = []
            try:
                out.append(MissingFeature(
                    feature=str(item.get("feature", ""))[:120],
                    requested_by_n=int(item.get("requested_by_n", 1)),
                    segments_requesting=[str(s)[:80] for s in segs_raw if s][:6],
                    example_request=str(item.get("example_request", ""))[:200],
                ))
            except (ValidationError, ValueError, TypeError):
                continue
        return out

    def _verify_hypotheses(
        self, ontology: Ontology, thread: ForumThread
    ) -> list[HypothesisVerdict]:
        if not ontology.user_hypotheses:
            return []
        hypotheses_json = json.dumps(
            [h.model_dump() for h in ontology.user_hypotheses], indent=2, ensure_ascii=False
        )
        # Compact forum content — key signals per post
        lines = []
        for p in thread.posts:
            flags: list[str] = []
            if p.would_pay != "unspecified":
                flags.append(f"pay={p.would_pay}")
            if p.biggest_objection:
                flags.append(f'obj="{p.biggest_objection}"')
            if p.wants_feature:
                flags.append(f'wants="{p.wants_feature}"')
            if p.final_verdict != "unspecified":
                flags.append(f"verdict={p.final_verdict}")
            flags_str = f" [{', '.join(flags)}]" if flags else ""
            lines.append(f"- [{p.id}, {p.sentiment}]{flags_str} {p.content[:200]}")
        forum_content = "\n".join(lines[:40])

        prompt = _HYPOTHESES_TEMPLATE.format(
            hypotheses_json=hypotheses_json,
            n_posts=len(thread.posts),
            forum_content=forum_content,
        )
        data = self._chat_json(
            system=_HYPOTHESES_SYSTEM,
            prompt=prompt,
            model=self.cluster_model,
            phase="hypotheses",
            max_tokens=2500,
        )
        if data is None:
            return []
        raw_items = data.get("hypotheses_verdict") or []
        out: list[HypothesisVerdict] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                out.append(HypothesisVerdict(
                    statement=str(item.get("statement", ""))[:400],
                    verdict=_coerce_verdict(item.get("verdict")),
                    evidence=str(item.get("evidence", ""))[:600],
                    confidence=_coerce_confidence(item.get("confidence")),
                ))
            except (ValidationError, ValueError, TypeError):
                continue
        return out

    def _synthesize(
        self,
        brief: str,
        ontology: Ontology,
        panel_composition: dict[str, int],
        adoption: list[SegmentAdoption],
        objections: list[ReportObjection],
        missing: list[MissingFeature],
        switching: list[SwitchingIntent],
        pricing: PricingFeedback,
        hypotheses: list[HypothesisVerdict],
        thread: ForumThread,
    ) -> dict[str, Any]:
        aggregated = {
            "adoption_by_segment": [a.model_dump() for a in adoption],
            "top_objections": [o.model_dump() for o in objections],
            "missing_features": [m.model_dump() for m in missing],
            "switching_intent": [s.model_dump() for s in switching],
            "pricing_feedback": pricing.model_dump(),
            "hypotheses_verdict": [h.model_dump() for h in hypotheses],
        }
        # Most-engaged r1 posts
        r1 = thread.posts_by_round(1)
        r1_sorted = sorted(
            r1,
            key=lambda p: thread.likes_for(p.id)
            + len([c for c in thread.comments if c.parent_post_id == p.id]),
            reverse=True,
        )
        highlights_lines = [
            f"- [r1, {p.sentiment}, {thread.likes_for(p.id)}❤] {p.content[:200]}"
            for p in r1_sorted[:5]
        ]
        # A few round-3 verdicts
        for p in thread.posts_by_round(3)[:4]:
            highlights_lines.append(
                f"- [r3, verdict={p.final_verdict}, pay={p.would_pay}] {p.content[:200]}"
            )
        highlights = "\n".join(highlights_lines)

        prompt = _SYNTHESIS_TEMPLATE.format(
            brief=brief[:2000],
            ontology_summary=ontology.analysis_summary[:400],
            panel_composition=json.dumps(panel_composition, ensure_ascii=False),
            aggregated=json.dumps(aggregated, indent=2, ensure_ascii=False)[:6000],
            highlights=highlights,
        )
        data = self._chat_json(
            system=_SYNTHESIS_SYSTEM,
            prompt=prompt,
            model=self.synthesis_model,
            phase="synthesis",
            max_tokens=2000,
        )
        if data is None:
            return {
                "brief_summary": "(synthesis call failed)",
                "red_flags": [],
                "recommended_mvp_cuts": [],
                "go_no_go_recommendation": "pivot",
                "go_no_go_rationale": "Synthesis call failed — review raw aggregated data.",
            }
        data["go_no_go_recommendation"] = _coerce_go_no_go(data.get("go_no_go_recommendation"))
        return data


# =========================================================================
# Coercion helpers for LLM enum outputs
# =========================================================================


def _coerce_severity(value: Any) -> str:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("blocker", "friction", "minor"):
            return v
    return "friction"


def _coerce_verdict(value: Any) -> str:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("validated", "invalidated", "inconclusive"):
            return v
    return "inconclusive"


def _coerce_confidence(value: Any) -> str:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("high", "medium", "low"):
            return v
    return "low"


def _coerce_go_no_go(value: Any) -> str:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("go", "pivot", "kill"):
            return v
    return "pivot"


def _panel_composition(panel: list[Persona]) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in panel:
        out[p.segment_name] = out.get(p.segment_name, 0) + 1
    return out

"""Zep Cloud knowledge-graph memory for PreFlight runs.

Each run's artefacts (brief, ontology, personas, forum verdicts, validation
report) are pushed as episodes to a single project-scoped graph. Zep extracts
entities (Persona, Segment, Competitor, Concern, Feature…) and builds a
temporal graph we can search across runs.

Design choices:
- One graph per PreFlight install (`graph_id="preflight"`), not per user. This
  is a product-validation tool, not a consumer chat app — memory is project-
  wide so queries like "which objections recur across briefs targeting
  freelancers?" work out of the box.
- Ingestion happens after `persist_run()` writes the local artefacts. Any Zep
  failure is logged + event-published, never raised — Zep is a sur-système,
  the JSON/Parquet on disk remain the source of truth.
- We batch episodes with `graph.add_batch()` (Zep processes them concurrently);
  one call per run instead of N round-trips.
- Idempotency is tracked locally in `data/runs/.zep_ingested.json`. Zep itself
  has entity-level dedup but not episode-level — re-ingesting the same run
  would produce duplicate facts.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import TYPE_CHECKING, Any

from zep_cloud import EpisodeData
from zep_cloud.client import Zep
from zep_cloud.core.api_error import ApiError

from sim_config import settings
from events import publish
from paths import RUNS_DIR

if TYPE_CHECKING:
    from app.services.pipeline import RunResult

logger = logging.getLogger(__name__)

# Episode cap per run: brief + ontology + ~20 personas + ~20 round-3 posts +
# report summary + judge ≈ 45 episodes max. Zep rate-limits batches; keep well
# under any plausible ceiling.
_MAX_EPISODES_PER_RUN = 80

_INGESTED_LEDGER = RUNS_DIR / ".zep_ingested.json"


class ZepMemory:
    """Thin wrapper around `zep_cloud.Zep`. Singleton-ish via `get_memory()`.

    Memory is scoped per-user: each authenticated identity gets its own
    Zep `user` (auto-created on first ingest) and the `graph.add_batch` /
    `graph.search` calls pass `user_id=...` instead of a project-global
    `graph_id`. Cross-run patterns stay private to the user that produced
    them. The dev-local fallback uses the configured `dev_user_id` as a
    Zep user name, so a single dev still gets a consistent memory across
    runs.
    """

    def __init__(self, api_key: str):
        self._client = Zep(api_key=api_key)
        # Tracks which Zep users we've already ensured this process —
        # avoids a redundant `user.add` round-trip on every ingest.
        self._users_ready: set[str] = set()
        self._lock = threading.Lock()

    # --- user bootstrap -------------------------------------------------

    def _ensure_user(self, user_id: str) -> None:
        """Create the Zep user on first reference. Idempotent."""
        with self._lock:
            if user_id in self._users_ready:
                return
            try:
                self._client.user.add(user_id=user_id)
                logger.info("zep: created user %s", user_id)
            except ApiError as e:
                # 400/409 = user already exists. Anything else propagates.
                if e.status_code in (400, 409):
                    logger.debug("zep: user %s already exists", user_id)
                else:
                    raise
            self._users_ready.add(user_id)

    # --- ingestion ------------------------------------------------------

    def ingest_run(self, result: "RunResult") -> int:
        """Push a RunResult to the user's graph. Returns the number of
        episodes sent.

        Never raises — any error is caught, logged, and a `zep.error`
        event is published. Callers should treat a 0 return as "ingestion
        failed or skipped" and keep going.
        """
        user_id = (result.user_id or "anon").strip() or "anon"

        if _already_ingested(result.run_id):
            logger.info("zep: run %s already ingested, skipping", result.run_id)
            publish("zep.skipped", {"run_id": result.run_id, "reason": "already_ingested"})
            return 0

        try:
            self._ensure_user(user_id)
            episodes = _build_episodes(result)
            if not episodes:
                return 0
            if len(episodes) > _MAX_EPISODES_PER_RUN:
                logger.warning(
                    "zep: run %s produced %d episodes, capping at %d",
                    result.run_id, len(episodes), _MAX_EPISODES_PER_RUN,
                )
                episodes = episodes[:_MAX_EPISODES_PER_RUN]

            publish(
                "zep.ingest_start",
                {"run_id": result.run_id, "n_episodes": len(episodes), "user_id": user_id},
            )
            self._client.graph.add_batch(episodes=episodes, user_id=user_id)
            _mark_ingested(result.run_id, n_episodes=len(episodes))
            publish(
                "zep.ingested",
                {"run_id": result.run_id, "n_episodes": len(episodes), "user_id": user_id},
            )
            logger.info(
                "zep: ingested %d episodes for run %s (user %s)",
                len(episodes), result.run_id, user_id,
            )
            return len(episodes)
        except Exception as e:  # noqa: BLE001 — we never let Zep fail a run
            logger.exception("zep: ingestion failed for run %s", result.run_id)
            publish("zep.error", {"run_id": result.run_id, "error": str(e)})
            return 0

    # --- query ----------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        user_id: str,
        scope: str = "edges",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search the *user's* graph. Returns raw edge/node dicts for the
        caller to format.

        Auto-creates the user before searching (defensive — a search on a
        not-yet-existing user returns 404 from Zep otherwise). For users
        with no ingested runs yet the result will be empty, which is the
        correct semantic: no memory to draw on.
        """
        self._ensure_user(user_id)
        try:
            results = self._client.graph.search(
                query=query,
                user_id=user_id,
                scope=scope,  # type: ignore[arg-type]
                limit=limit,
            )
        except ApiError as e:
            # If the user has truly no graph yet (404) or the query lands
            # on an empty store, return [] rather than bubbling.
            if e.status_code in (404,):
                return []
            raise
        # GraphSearchResults has .edges / .nodes; we expose whichever scope asked for.
        if scope == "nodes":
            return [n.dict() if hasattr(n, "dict") else dict(n) for n in (results.nodes or [])]
        return [e.dict() if hasattr(e, "dict") else dict(e) for e in (results.edges or [])]


# --- module-level singleton + accessors --------------------------------

_memory: ZepMemory | None = None
_memory_lock = threading.Lock()


def get_memory() -> ZepMemory | None:
    """Return the shared ZepMemory instance, or None if ZEP_API_KEY is unset."""
    global _memory
    if _memory is not None:
        return _memory
    with _memory_lock:
        if _memory is not None:
            return _memory
        key = settings().zep_api_key.strip()
        if not key:
            return None
        _memory = ZepMemory(api_key=key)
        return _memory


def is_enabled() -> bool:
    return bool(settings().zep_api_key.strip())


# --- episode builders --------------------------------------------------

def _build_episodes(result: "RunResult") -> list[EpisodeData]:
    """Break a RunResult into the episodes we want Zep to ingest.

    Strategy: one episode per semantically distinct unit (brief, ontology,
    each persona, each verdict-bearing post, report summary). Keeps each
    payload small enough for Zep's entity extractor to stay focused.
    """
    episodes: list[EpisodeData] = []
    rid = result.run_id
    tag = lambda kind, ident=None: (  # noqa: E731
        f"preflight:run:{rid}:{kind}" + (f":{ident}" if ident else "")
    )

    # 1. Brief — text, unstructured
    episodes.append(
        EpisodeData(
            data=result.brief,
            type="text",
            source_description=tag("brief"),
        )
    )

    # 2. Ontology summary — the structured spine of what was tested
    ont = result.ontology
    episodes.append(
        EpisodeData(
            data=json.dumps(
                {
                    "run_id": rid,
                    "kind": "ontology",
                    "segments": [
                        {"name": s.name, "pain_level": s.pain_level,
                         "description": s.description}
                        for s in ont.segments
                    ],
                    "competitors": [
                        {"name": c.name, "category": c.category,
                         "why_users_use_it": c.why_users_use_it}
                        for c in ont.competitors
                    ],
                    "features": [
                        {"name": f.name, "priority": f.priority_in_brief,
                         "answers_pain": f.answers_pain}
                        for f in ont.features
                    ],
                    "hypotheses": [
                        {"statement": h.statement, "metric": h.testable_metric,
                         "strength": h.strength}
                        for h in ont.user_hypotheses
                    ],
                },
                ensure_ascii=False,
            ),
            type="json",
            source_description=tag("ontology"),
        )
    )

    # 3. Each persona — rich entity for the graph (name + segment + pain + stack)
    for p in result.panel:
        episodes.append(
            EpisodeData(
                data=json.dumps(
                    {
                        "run_id": rid,
                        "kind": "persona",
                        "persona_id": p.id,
                        "name": p.name,
                        "age": p.age,
                        "role": p.role,
                        "location": p.location,
                        "segment": p.segment_name,
                        "current_pain": p.current_pain,
                        "current_stack": p.current_stack,
                        "tech_attitude": p.tech_attitude,
                        "willingness_to_pay_eur_per_month": p.willingness_to_pay_eur_per_month,
                    },
                    ensure_ascii=False,
                ),
                type="json",
                source_description=tag("persona", p.id),
            )
        )

    # 4. Verdict-bearing posts — round-3 posts have final_verdict + structured
    # signals populated. Rounds 1-2 are chatter; skipping them keeps graph
    # signal/noise high.
    verdict_posts = [
        post for post in result.thread.posts
        if post.round == result.rounds and post.final_verdict != "unspecified"
    ]
    for post in verdict_posts:
        episodes.append(
            EpisodeData(
                data=json.dumps(
                    {
                        "run_id": rid,
                        "kind": "verdict",
                        "post_id": post.id,
                        "persona_id": post.persona_id,
                        "final_verdict": post.final_verdict,
                        "would_pay": post.would_pay,
                        "biggest_objection": post.biggest_objection,
                        "wants_feature": post.wants_feature,
                        "switch_from": post.switch_from,
                        "sentiment": post.sentiment,
                        "quote": post.content,
                    },
                    ensure_ascii=False,
                ),
                type="json",
                source_description=tag("verdict", post.id),
            )
        )

    # 5. Validation report digest — the final clustered view
    report = result.validation_report
    episodes.append(
        EpisodeData(
            data=json.dumps(
                {
                    "run_id": rid,
                    "kind": "validation_report",
                    "go_no_go": report.go_no_go_recommendation,
                    "rationale": report.go_no_go_rationale,
                    "adoption_by_segment": [
                        {"segment": a.segment_name, "score": a.adoption_score,
                         "supporters": a.n_supporters, "detractors": a.n_detractors}
                        for a in report.adoption_by_segment
                    ],
                    "top_objections": [
                        {"text": o.text, "frequency": o.frequency, "severity": o.severity}
                        for o in report.top_objections[:5]
                    ],
                    "missing_features": [
                        {"feature": m.feature, "requested_by_n": m.requested_by_n,
                         "segments": m.segments_requesting}
                        for m in report.missing_features[:5]
                    ],
                    "pricing": {
                        "floor_eur": report.pricing_feedback.floor_eur_month,
                        "ceiling_eur": report.pricing_feedback.ceiling_eur_month,
                    },
                    "hypotheses_verdict": [
                        {"statement": h.statement, "verdict": h.verdict,
                         "confidence": h.confidence}
                        for h in report.hypotheses_verdict
                    ],
                    "red_flags": report.red_flags,
                },
                ensure_ascii=False,
            ),
            type="json",
            source_description=tag("validation_report"),
        )
    )

    return episodes


# --- local ledger (idempotency) ----------------------------------------

def _load_ledger() -> dict[str, Any]:
    if not _INGESTED_LEDGER.exists():
        return {}
    try:
        return json.loads(_INGESTED_LEDGER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("zep: ledger unreadable, treating as empty")
        return {}


def _save_ledger(data: dict[str, Any]) -> None:
    _INGESTED_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    tmp = _INGESTED_LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(_INGESTED_LEDGER)


def _already_ingested(run_id: str) -> bool:
    return run_id in _load_ledger()


def _mark_ingested(run_id: str, *, n_episodes: int) -> None:
    ledger = _load_ledger()
    ledger[run_id] = {"n_episodes": n_episodes}
    _save_ledger(ledger)


def forget_run(run_id: str) -> None:
    """Remove a run from the local ledger so re-ingestion is allowed. Does
    NOT delete episodes already in Zep — use the Zep dashboard for that."""
    ledger = _load_ledger()
    if run_id in ledger:
        del ledger[run_id]
        _save_ledger(ledger)


__all__ = [
    "ZepMemory",
    "get_memory",
    "is_enabled",
    "forget_run",
]

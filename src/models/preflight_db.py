"""Postgres repository for PreFlight runs + run_artifacts (BE-PR1).

Wraps psycopg3 calls against the Drizzle-managed schema landed in the
sister frontend project (see migrations/0000_*.sql there). The frontend
defines + migrates the schema; this module only reads + writes against
it.

Design choices:

- **Per-call connection.** psycopg connections aren't thread-safe and
  the existing `models/pgsql/client.py` exposes a single shared one
  that would race under concurrent run-ingestions / chat persistence.
  Each public function here opens its own short-lived connection. The
  per-call setup cost (~10 ms) is invisible next to LLM round-trips.

- **Graceful no-op.** Every public function returns False / None / []
  when DATABASE_URL is missing OR the auth_uid isn't a real UUID
  (dev-local mode uses literal "dev-local" which isn't a Postgres
  uuid). Callers (pipeline.persist_run, endpoints/runs.py,
  endpoints/chat.py) treat that as "DB unavailable, fall back to
  file mode" — keeps the legacy data/runs/{user_id}/*.json path
  alive for laptop dev.

- **Auto-upsert users row.** Supabase auth creates the `auth.users`
  entry but our app-side `users` table needs its own row for the FK
  link. First time we see an auth_uid, we INSERT a stub row
  (username + email derived from the uid). The friend's signup flow
  on the frontend can do the same INSERT; ours is a defensive
  belt-and-braces so a backend-first dev path works.

Schema reference (Drizzle, frontend-side):
    runs(id uuid PK, user_id int FK users.id, brief text, panel_size int,
         rounds int, settings jsonb, status run_status, verdict run_verdict,
         cost_usd numeric, wall_s numeric, rationale text, error_message text,
         started_at timestamp, completed_at timestamp)
    run_artifacts(id uuid PK, run_id uuid FK runs.id ON DELETE CASCADE,
                  kind run_artefact_kind, payload jsonb,
                  created_at timestamp, updated_at timestamp)
                  UNIQUE(run_id, kind)
"""
from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager
from os import environ as env
from typing import Any, Iterator
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)


# ---- Connection helpers ---------------------------------------------------

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _is_uuid(s: str) -> bool:
    return bool(_UUID_RE.match(s))


def is_db_available() -> bool:
    """True iff DATABASE_URL is set. Doesn't actually connect — caller
    can attempt a connection with @contextmanager `connect()` and handle
    failures there."""
    return bool(env.get("DATABASE_URL"))


@contextmanager
def connect() -> Iterator[psycopg.Connection | None]:
    """Per-call connection. Yields None when no DATABASE_URL or the connect
    fails, so callers can branch on `if conn is None: fall back to files`.

    autocommit=True so simple INSERT/SELECT statements don't need explicit
    BEGIN/COMMIT. The longer write-paths (persist_run with multiple
    artefact UPSERTs) wrap their own transaction via a `with conn.transaction()`
    block.
    """
    url = env.get("DATABASE_URL")
    if not url:
        yield None
        return
    conn: psycopg.Connection | None = None
    try:
        conn = psycopg.connect(url, autocommit=True)
        yield conn
    except Exception as e:  # noqa: BLE001
        logger.warning("preflight_db: connect failed — %s", e)
        yield None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass


# ---- User-id resolution ---------------------------------------------------

def _resolve_user_pk(conn: psycopg.Connection, auth_uid: str) -> int | None:
    """Map a Supabase auth.uid (uuid string) → users.id (integer PK).

    Auto-creates the users row on first sight so a backend-first request
    flow (e.g. CLI, or a frontend that hasn't hit /api-client/auth/signup)
    still has somewhere to attach runs. Returns None when auth_uid isn't
    a valid UUID — the caller treats that as "fall back to file mode".
    """
    if not _is_uuid(auth_uid):
        return None
    cursor = conn.execute("SELECT id FROM users WHERE auth_id = %s", (auth_uid,))
    row = cursor.fetchone()
    if row:
        return int(row[0])
    # Stub row — username + email are placeholders the user can edit
    # later. plan defaults to "free" (matches the schema default).
    short = auth_uid[:8]
    insert = conn.execute(
        """
        INSERT INTO users (auth_id, username, email)
        VALUES (%s, %s, %s)
        ON CONFLICT (auth_id) DO UPDATE SET auth_id = EXCLUDED.auth_id
        RETURNING id
        """,
        (auth_uid, f"user_{short}", f"{auth_uid}@auth.local"),
    )
    new_row = insert.fetchone()
    return int(new_row[0]) if new_row else None


# ---- Runs -----------------------------------------------------------------

def insert_run(
    *,
    run_id: str,
    auth_uid: str,
    brief: str,
    panel_size: int,
    rounds: int,
    settings: dict[str, Any],
) -> bool:
    """Create the `runs` row at pipeline start. Returns True on success,
    False when the DB is unavailable or auth_uid is not a UUID (callers
    fall back to file mode).
    """
    with connect() as conn:
        if conn is None:
            return False
        try:
            user_pk = _resolve_user_pk(conn, auth_uid)
            if user_pk is None:
                return False
            conn.execute(
                """
                INSERT INTO runs (id, user_id, brief, panel_size, rounds, settings, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'running')
                ON CONFLICT (id) DO NOTHING
                """,
                (run_id, user_pk, brief, panel_size, rounds, Jsonb(settings)),
            )
            return True
        except Exception:  # noqa: BLE001
            logger.exception("preflight_db.insert_run failed")
            return False


def update_run_terminal(
    *,
    run_id: str,
    status: str,  # "done" | "error"
    verdict: str | None = None,  # "go" | "pivot" | "kill" | None
    cost_usd: float | None = None,
    wall_s: float | None = None,
    rationale: str | None = None,
    error_message: str | None = None,
) -> bool:
    """Stamp the terminal state when a run finishes (success or failure).

    Gated on `status='running'` so a late worker-thread completion can't
    overwrite a cancellation or the orphan-recovery sweep — once a row
    leaves the running state, it stays put.
    """
    if status not in ("done", "error"):
        raise ValueError(f"unsupported terminal status: {status}")
    with connect() as conn:
        if conn is None:
            return False
        try:
            conn.execute(
                """
                UPDATE runs
                SET status = %s,
                    verdict = %s,
                    cost_usd = %s,
                    wall_s = %s,
                    rationale = %s,
                    error_message = %s,
                    completed_at = NOW()
                WHERE id = %s AND status = 'running'
                """,
                (status, verdict, cost_usd, wall_s, rationale, error_message, run_id),
            )
            return True
        except Exception:  # noqa: BLE001
            logger.exception("preflight_db.update_run_terminal failed")
            return False


def delete_run(*, run_id: str, auth_uid: str) -> str | None:
    """Hard-delete a run owned by this user. Refuses in-flight runs.

    Returns:
        "deleted"   — row gone, run_artifacts cascaded via FK
        "not_found" — no such run for this user
        "running"   — refused; caller should /cancel first
        None        — DB unavailable / not a UUID auth_uid

    Two queries (status probe + DELETE) keeps the three terminal cases
    distinguishable; rolling them into a single DELETE/RETURNING would
    collapse "not_found" and "running" into the same empty result.
    """
    with connect() as conn:
        if conn is None:
            return None
        try:
            user_pk = _resolve_user_pk(conn, auth_uid)
            if user_pk is None:
                return None
            row = conn.execute(
                "SELECT status::text FROM runs WHERE id = %s AND user_id = %s",
                (run_id, user_pk),
            ).fetchone()
            if not row:
                return "not_found"
            if row[0] == "running":
                return "running"
            conn.execute(
                "DELETE FROM runs WHERE id = %s AND user_id = %s",
                (run_id, user_pk),
            )
            return "deleted"
        except Exception:  # noqa: BLE001
            logger.exception("preflight_db.delete_run failed")
            return None


def cancel_run(*, run_id: str, auth_uid: str) -> bool | None:
    """Mark a user's in-flight run as error('cancelled by user').

    Returns True if a row was actually transitioned, False if the run
    didn't exist or wasn't in status='running' for this user, or None
    when the DB is unavailable. The worker thread keeps running but
    update_run_terminal is gated on status='running' so its eventual
    output won't overwrite the cancelled state.
    """
    with connect() as conn:
        if conn is None:
            return None
        try:
            user_pk = _resolve_user_pk(conn, auth_uid)
            if user_pk is None:
                return None
            cur = conn.execute(
                """
                UPDATE runs
                SET status = 'error',
                    error_message = 'cancelled by user',
                    completed_at = NOW()
                WHERE id = %s AND user_id = %s AND status = 'running'
                """,
                (run_id, user_pk),
            )
            return (cur.rowcount or 0) > 0
        except Exception:  # noqa: BLE001
            logger.exception("preflight_db.cancel_run failed")
            return None


def upsert_artefact(*, run_id: str, kind: str, payload: dict[str, Any]) -> bool:
    """Store (or replace) a kind-tagged artefact row.

    `kind` must match the run_artefact_kind enum: ontology | panel |
    thread | validation_report | judge_scores | chat_history.
    """
    with connect() as conn:
        if conn is None:
            return False
        try:
            conn.execute(
                """
                INSERT INTO run_artifacts (run_id, kind, payload)
                VALUES (%s, %s, %s)
                ON CONFLICT (run_id, kind)
                DO UPDATE SET payload = EXCLUDED.payload, updated_at = NOW()
                """,
                (run_id, kind, Jsonb(payload)),
            )
            return True
        except Exception:  # noqa: BLE001
            logger.exception("preflight_db.upsert_artefact failed (kind=%s)", kind)
            return False


# ---- Diagnostics ----------------------------------------------------------

_REQUIRED_TABLES = ("runs", "run_artifacts", "users")


def schema_check() -> dict[str, Any]:
    """Probe the DB and report which preflight tables are reachable.

    Returns a flat dict suitable for logging or surfacing through /health:

        {
            "database_url_set": bool,
            "connection_ok": bool,
            "tables_present": {"runs": bool, "run_artifacts": bool, "users": bool},
            "schema_ok": bool,           # all required tables present
        }

    Never raises — a closed/missing DB collapses to all-False.
    """
    out: dict[str, Any] = {
        "database_url_set": bool(env.get("DATABASE_URL")),
        "connection_ok": False,
        "tables_present": {t: False for t in _REQUIRED_TABLES},
        "schema_ok": False,
    }
    if not out["database_url_set"]:
        return out
    with connect() as conn:
        if conn is None:
            return out
        out["connection_ok"] = True
        try:
            cur = conn.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = ANY(%s)
                """,
                (list(_REQUIRED_TABLES),),
            )
            present = {row[0] for row in cur.fetchall()}
            for t in _REQUIRED_TABLES:
                out["tables_present"][t] = t in present
            out["schema_ok"] = all(out["tables_present"].values())
        except Exception:  # noqa: BLE001
            logger.exception("preflight_db.schema_check failed")
    return out


# ---- Lifecycle housekeeping -----------------------------------------------

def recover_orphan_runs() -> int:
    """Mark every status='running' row as error.

    Called once at FastAPI startup. The pipeline only ever writes
    status='running' on insert and updates terminal via
    update_run_terminal. So a row stuck at 'running' at boot means the
    worker thread that owned it didn't survive — without this sweep, the
    user stays locked out by the per-user concurrency check (BE-PR3).

    Single-replica assumption: in a multi-replica deployment this would
    kill runs in flight on sibling processes. Add a heartbeat / replica
    id before scaling out.
    """
    with connect() as conn:
        if conn is None:
            return 0
        try:
            cur = conn.execute(
                """
                UPDATE runs
                SET status = 'error',
                    error_message = COALESCE(error_message, 'abandoned: server restart'),
                    completed_at = NOW()
                WHERE status = 'running'
                """,
            )
            return cur.rowcount or 0
        except Exception:  # noqa: BLE001
            logger.exception("preflight_db.recover_orphan_runs failed")
            return 0


# ---- Concurrency ----------------------------------------------------------

def has_running_run_for_user(auth_uid: str) -> bool | None:
    """Returns True iff this user already has a run row in status='running'.

    None when DB is unavailable or auth_uid isn't a UUID — caller falls
    back to the in-memory global lock so dev-local keeps working without
    a DATABASE_URL.
    """
    with connect() as conn:
        if conn is None:
            return None
        try:
            user_pk = _resolve_user_pk(conn, auth_uid)
            if user_pk is None:
                return None
            row = conn.execute(
                """
                SELECT 1 FROM runs
                WHERE user_id = %s AND status = 'running'
                LIMIT 1
                """,
                (user_pk,),
            ).fetchone()
            return row is not None
        except Exception:  # noqa: BLE001
            logger.exception("preflight_db.has_running_run_for_user failed")
            return None


# ---- Reads ----------------------------------------------------------------

def list_runs_for_user(
    auth_uid: str,
    *,
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
) -> list[dict[str, Any]] | None:
    """Sidebar feed: newest-first page of runs owned by this user.
    Returns None when DB is unavailable so callers can fall back to the
    on-disk glob. Ordered by `started_at` DESC; the frontend treats
    each request as a page (offset is bytes-old-style, not a cursor —
    fine for hackathon scale, swap to keyset before scaling out).

    `status` optionally filters to one of the run_status enum values
    (running|done|error). The endpoint validates against the enum
    before passing through.
    """
    with connect() as conn:
        if conn is None:
            return None
        try:
            user_pk = _resolve_user_pk(conn, auth_uid)
            if user_pk is None:
                return None
            params: list[Any] = [user_pk]
            status_clause = ""
            if status is not None:
                status_clause = " AND status = %s"
                params.append(status)
            params.extend([limit, offset])
            cur = conn.execute(
                f"""
                SELECT id::text,
                       brief,
                       panel_size,
                       rounds,
                       status::text,
                       verdict::text,
                       cost_usd,
                       wall_s,
                       rationale,
                       error_message,
                       started_at,
                       completed_at
                FROM runs
                WHERE user_id = %s{status_clause}
                ORDER BY started_at DESC
                LIMIT %s OFFSET %s
                """,
                tuple(params),
            )
            cols = [d.name for d in (cur.description or [])]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception:  # noqa: BLE001
            logger.exception("preflight_db.list_runs_for_user failed")
            return None


def user_run_stats(auth_uid: str) -> dict[str, Any] | None:
    """Aggregate stats for the dashboard. None when DB is unavailable.

    Returned shape (always concrete keys, zeroed when no rows match):

        {
            "total": 12,
            "by_status": {"running": 1, "done": 10, "error": 1},
            "by_verdict": {"go": 7, "pivot": 2, "kill": 1, "unknown": 2},
            "total_cost_usd": 4.5670,
            "first_run_at": "2026-04-12T...",
            "last_run_at":  "2026-04-26T...",
        }

    Single round-trip: one SELECT with FILTER aggregates instead of N
    GROUP BYs. Verdict bucket "unknown" covers rows where the verdict
    column is NULL (errored runs and very-fresh ones).
    """
    with connect() as conn:
        if conn is None:
            return None
        try:
            user_pk = _resolve_user_pk(conn, auth_uid)
            if user_pk is None:
                return None
            row = conn.execute(
                """
                SELECT
                    COUNT(*)                                                AS total,
                    COUNT(*) FILTER (WHERE status = 'running')              AS s_running,
                    COUNT(*) FILTER (WHERE status = 'done')                 AS s_done,
                    COUNT(*) FILTER (WHERE status = 'error')                AS s_error,
                    COUNT(*) FILTER (WHERE verdict = 'go')                  AS v_go,
                    COUNT(*) FILTER (WHERE verdict = 'pivot')               AS v_pivot,
                    COUNT(*) FILTER (WHERE verdict = 'kill')                AS v_kill,
                    COUNT(*) FILTER (WHERE verdict IS NULL)                 AS v_unknown,
                    COALESCE(SUM(cost_usd), 0)                              AS total_cost,
                    MIN(started_at)                                         AS first_at,
                    MAX(started_at)                                         AS last_at
                FROM runs
                WHERE user_id = %s
                """,
                (user_pk,),
            ).fetchone()
            if not row:
                return _empty_stats()
            (
                total, sr, sd, se, vg, vp, vk, vu, total_cost, first_at, last_at,
            ) = row
            return {
                "total": int(total or 0),
                "by_status": {
                    "running": int(sr or 0),
                    "done": int(sd or 0),
                    "error": int(se or 0),
                },
                "by_verdict": {
                    "go": int(vg or 0),
                    "pivot": int(vp or 0),
                    "kill": int(vk or 0),
                    "unknown": int(vu or 0),
                },
                "total_cost_usd": float(total_cost or 0),
                "first_run_at": first_at.isoformat() if first_at else None,
                "last_run_at": last_at.isoformat() if last_at else None,
            }
        except Exception:  # noqa: BLE001
            logger.exception("preflight_db.user_run_stats failed")
            return None


def _empty_stats() -> dict[str, Any]:
    return {
        "total": 0,
        "by_status": {"running": 0, "done": 0, "error": 0},
        "by_verdict": {"go": 0, "pivot": 0, "kill": 0, "unknown": 0},
        "total_cost_usd": 0.0,
        "first_run_at": None,
        "last_run_at": None,
    }


def get_run_with_artifacts(*, run_id: str, auth_uid: str) -> dict[str, Any] | None:
    """Return the run row + every artefact (keyed by kind) the user owns
    for this id. None when not found OR not owned by this user OR DB
    unavailable.
    """
    with connect() as conn:
        if conn is None:
            return None
        try:
            user_pk = _resolve_user_pk(conn, auth_uid)
            if user_pk is None:
                return None
            cur = conn.execute(
                """
                SELECT id::text, brief, panel_size, rounds, settings,
                       status::text, verdict::text, cost_usd, wall_s,
                       rationale, error_message, started_at, completed_at
                FROM runs
                WHERE id = %s AND user_id = %s
                """,
                (run_id, user_pk),
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [d.name for d in (cur.description or [])]
            run = dict(zip(cols, row))

            cur = conn.execute(
                """
                SELECT kind::text, payload, created_at, updated_at
                FROM run_artifacts
                WHERE run_id = %s
                """,
                (run_id,),
            )
            artefacts: dict[str, Any] = {}
            for kind, payload, created_at, updated_at in cur.fetchall():
                artefacts[kind] = {
                    "payload": payload,
                    "created_at": created_at,
                    "updated_at": updated_at,
                }
            run["artifacts"] = artefacts
            return run
        except Exception:  # noqa: BLE001
            logger.exception("preflight_db.get_run_with_artifacts failed")
            return None


def get_artefact(
    *, run_id: str, kind: str, auth_uid: str,
) -> dict[str, Any] | None:
    """Fetch one artefact for a run owned by this user.

    Returns:
        {"kind": ..., "payload": ..., "created_at": ..., "updated_at": ...}
        — or None when not found OR not owned OR DB unavailable.

    The caller is expected to validate `kind` against the artefact_kind
    enum (the endpoint does this via a whitelist) — Postgres would
    happily accept anything castable to the enum, but bad input should
    400 earlier.
    """
    with connect() as conn:
        if conn is None:
            return None
        try:
            user_pk = _resolve_user_pk(conn, auth_uid)
            if user_pk is None:
                return None
            row = conn.execute(
                """
                SELECT a.kind::text, a.payload, a.created_at, a.updated_at
                FROM run_artifacts a
                JOIN runs r ON r.id = a.run_id
                WHERE a.run_id = %s AND a.kind = %s AND r.user_id = %s
                """,
                (run_id, kind, user_pk),
            ).fetchone()
            if not row:
                return None
            kind_text, payload, created_at, updated_at = row
            return {
                "kind": kind_text,
                "payload": payload,
                "created_at": created_at.isoformat() if created_at else None,
                "updated_at": updated_at.isoformat() if updated_at else None,
            }
        except Exception:  # noqa: BLE001
            logger.exception("preflight_db.get_artefact failed")
            return None


def get_chat_history(*, run_id: str, auth_uid: str) -> list[dict[str, Any]] | None:
    """Return the messages list stored under run_artifacts kind=chat_history,
    or [] when none yet, or None when DB is unavailable.
    """
    with connect() as conn:
        if conn is None:
            return None
        try:
            user_pk = _resolve_user_pk(conn, auth_uid)
            if user_pk is None:
                return None
            # Sub-select on run ownership so we don't leak across users.
            cur = conn.execute(
                """
                SELECT a.payload
                FROM run_artifacts a
                JOIN runs r ON r.id = a.run_id
                WHERE a.run_id = %s
                  AND a.kind = 'chat_history'
                  AND r.user_id = %s
                """,
                (run_id, user_pk),
            )
            row = cur.fetchone()
            if not row:
                return []
            payload = row[0] or {}
            msgs = payload.get("messages") if isinstance(payload, dict) else None
            return msgs if isinstance(msgs, list) else []
        except Exception:  # noqa: BLE001
            logger.exception("preflight_db.get_chat_history failed")
            return None


def upsert_chat_history(
    *,
    run_id: str,
    auth_uid: str,
    messages: list[dict[str, Any]],
) -> bool:
    """Replace the chat_history artefact wholesale. Each chat turn calls
    this with the full transcript so a refresh sees the same conversation
    state the user just produced.
    """
    with connect() as conn:
        if conn is None:
            return False
        try:
            user_pk = _resolve_user_pk(conn, auth_uid)
            if user_pk is None:
                return False
            # Defensive: only update chat_history for runs the caller owns.
            owns = conn.execute(
                "SELECT 1 FROM runs WHERE id = %s AND user_id = %s",
                (run_id, user_pk),
            ).fetchone()
            if not owns:
                return False
            conn.execute(
                """
                INSERT INTO run_artifacts (run_id, kind, payload)
                VALUES (%s, 'chat_history', %s)
                ON CONFLICT (run_id, kind)
                DO UPDATE SET payload = EXCLUDED.payload, updated_at = NOW()
                """,
                (run_id, Jsonb({"messages": messages})),
            )
            return True
        except Exception:  # noqa: BLE001
            logger.exception("preflight_db.upsert_chat_history failed")
            return False


__all__ = [
    "is_db_available",
    "schema_check",
    "connect",
    "insert_run",
    "update_run_terminal",
    "cancel_run",
    "delete_run",
    "upsert_artefact",
    "recover_orphan_runs",
    "has_running_run_for_user",
    "list_runs_for_user",
    "user_run_stats",
    "get_artefact",
    "get_run_with_artifacts",
    "get_chat_history",
    "upsert_chat_history",
]

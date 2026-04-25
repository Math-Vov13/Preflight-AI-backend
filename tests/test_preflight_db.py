"""preflight_db smoke tests — no live DB required.

These pin the graceful-fallback contract: every public function returns
None / False / an empty dict when DATABASE_URL is unset, so the
file-mode fallbacks in endpoints/runs.py + endpoints/chat.py keep
working without surprise side effects.
"""
from __future__ import annotations

import pytest

from models import preflight_db


@pytest.mark.parametrize(
    ("uid", "ok"),
    [
        ("550e8400-e29b-41d4-a716-446655440000", True),
        ("550E8400-E29B-41D4-A716-446655440000", True),
        ("dev-local", False),
        ("", False),
        ("not-a-uuid", False),
        # Right shape, bad chars
        ("zzzzzzzz-e29b-41d4-a716-446655440000", False),
    ],
)
def test_is_uuid_classification(uid: str, ok: bool) -> None:
    assert preflight_db._is_uuid(uid) is ok  # noqa: SLF001


def test_is_db_available_without_url() -> None:
    # autouse `isolate_env` strips DATABASE_URL.
    assert preflight_db.is_db_available() is False


def test_schema_check_collapses_to_all_false_without_url() -> None:
    snap = preflight_db.schema_check()
    assert snap == {
        "database_url_set": False,
        "connection_ok": False,
        "tables_present": {"runs": False, "run_artifacts": False, "users": False},
        "schema_ok": False,
    }


def test_no_db_writes_return_falsey() -> None:
    """Every write/read function must no-op when the DB is unreachable."""
    assert preflight_db.insert_run(
        run_id="ignored",
        auth_uid="dev-local",
        brief="x",
        panel_size=5,
        rounds=2,
        settings={},
    ) is False
    assert preflight_db.update_run_terminal(
        run_id="ignored", status="done",
    ) is False
    assert preflight_db.upsert_artefact(
        run_id="ignored", kind="ontology", payload={},
    ) is False
    assert preflight_db.recover_orphan_runs() == 0
    assert preflight_db.has_running_run_for_user("dev-local") is None
    assert preflight_db.list_runs_for_user("dev-local") is None
    assert preflight_db.get_run_with_artifacts(run_id="x", auth_uid="dev-local") is None
    assert preflight_db.get_chat_history(run_id="x", auth_uid="dev-local") is None
    assert preflight_db.upsert_chat_history(
        run_id="x", auth_uid="dev-local", messages=[],
    ) is False
    assert preflight_db.cancel_run(run_id="x", auth_uid="dev-local") is None
    assert preflight_db.delete_run(run_id="x", auth_uid="dev-local") is None


def test_update_run_terminal_rejects_non_terminal_status() -> None:
    with pytest.raises(ValueError):
        preflight_db.update_run_terminal(run_id="x", status="running")

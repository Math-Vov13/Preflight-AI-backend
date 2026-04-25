"""auth.session_payload + whoami session block (BE-PR24)."""
from __future__ import annotations

import time

import pytest
from jose import jwt

import auth
from endpoints.preflight_auth import whoami

_TEST_SECRET = "test-secret"


def _make_jwt(*, sub: str = "user-uuid", exp_offset: int = 3600) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "sub": sub,
            "aud": "authenticated",
            "iat": now,
            "exp": now + exp_offset,
        },
        _TEST_SECRET,
        algorithm="HS256",
    )


@pytest.fixture
def supabase_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPABASE_JWT_SECRET", _TEST_SECRET)
    # sim_config keeps the parsed Settings in a module-level cache —
    # invalidate so the new env is picked up.
    import sim_config

    sim_config._settings = None  # noqa: SLF001
    yield
    sim_config._settings = None  # noqa: SLF001


def test_session_payload_dev_local_returns_none() -> None:
    # autouse isolate_env strips SUPABASE_JWT_SECRET → dev-local.
    assert auth.session_payload("Bearer anything") is None


def test_session_payload_decodes_valid_token(supabase_mode: None) -> None:
    token = _make_jwt(sub="abc-uuid", exp_offset=600)
    payload = auth.session_payload(f"Bearer {token}")
    assert payload is not None
    assert payload["sub"] == "abc-uuid"
    assert "exp" in payload
    assert "iat" in payload


def test_session_payload_expired_returns_none(supabase_mode: None) -> None:
    """A token decoded after expiry shouldn't 401 from /auth/whoami —
    the session block just drops, since CurrentUser already covers the
    auth gate."""
    token = _make_jwt(exp_offset=-10)
    assert auth.session_payload(f"Bearer {token}") is None


def test_whoami_supabase_includes_session_block(
    supabase_mode: None,
) -> None:
    token = _make_jwt(sub="abc-uuid", exp_offset=900)
    body = whoami("abc-uuid", authorization=f"Bearer {token}")
    assert body["mode"] == "supabase"
    assert "session" in body
    assert isinstance(body["session"]["expires_at"], int)
    assert isinstance(body["session"]["issued_at"], int)
    assert body["session"]["expires_at"] > body["session"]["issued_at"]


def test_whoami_dev_local_omits_session() -> None:
    body = whoami("dev-local")
    assert body["mode"] == "dev-local"
    assert "session" not in body

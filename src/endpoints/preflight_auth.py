"""Auth endpoints — surface whoami + auth mode to the frontend.

The frontend uses GET /api/auth/whoami both as a session probe (200 means the
JWT is still valid; 401 means we should redirect to the sign-in page) and as
a discovery endpoint for whether real auth is even on (dev-local mode).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from auth import CurrentUser, auth_mode
from models import preflight_db

router = APIRouter(tags=["auth"])


@router.get("/auth/whoami")
def whoami(user: CurrentUser) -> dict[str, Any]:
    """Session probe + profile fetch in one call.

    Always returns at least {user_id, mode}. When a Postgres row is
    reachable we additionally return username/email/plan so the frontend
    can render a profile dropdown without a second request. The profile
    fields silently drop when the DB is unavailable (dev-local) — the
    caller treats their absence as 'no profile yet'.
    """
    body: dict[str, Any] = {"user_id": user, "mode": auth_mode()}
    profile = preflight_db.get_user_profile(user)
    if profile is not None:
        body["profile"] = profile
    return body


@router.get("/auth/mode")
def mode() -> dict[str, str]:
    """Unauthenticated probe — lets the frontend decide whether to render
    the sign-in screen at all (dev-local mode skips it)."""
    return {"mode": auth_mode()}

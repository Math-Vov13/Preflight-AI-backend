"""Auth endpoints — surface whoami + auth mode to the frontend.

The frontend uses GET /api/auth/whoami both as a session probe (200 means the
JWT is still valid; 401 means we should redirect to the sign-in page) and as
a discovery endpoint for whether real auth is even on (dev-local mode).
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Header

from auth import CurrentUser, auth_mode, session_payload
from models import preflight_db

router = APIRouter(tags=["auth"])


@router.get("/auth/whoami")
def whoami(
    user: CurrentUser,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, Any]:
    """Session probe + profile fetch in one call.

    Always returns at least {user_id, mode}. When a Postgres row is
    reachable we additionally return username/email/plan so the frontend
    can render a profile dropdown without a second request. When running
    in supabase mode we also surface session.{expires_at, issued_at} so
    the frontend can warn before silent JWT timeout. Both blocks drop
    silently in dev-local — the caller treats absence as 'not applicable'.
    """
    body: dict[str, Any] = {"user_id": user, "mode": auth_mode()}
    payload = session_payload(authorization)
    if payload is not None:
        body["session"] = {
            "expires_at": payload.get("exp"),
            "issued_at": payload.get("iat"),
        }
    profile = preflight_db.get_user_profile(user)
    if profile is not None:
        body["profile"] = profile
    return body


@router.get("/auth/mode")
def mode() -> dict[str, str]:
    """Unauthenticated probe — lets the frontend decide whether to render
    the sign-in screen at all (dev-local mode skips it)."""
    return {"mode": auth_mode()}

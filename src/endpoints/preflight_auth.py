"""Auth endpoints — surface whoami + auth mode to the frontend.

The frontend uses GET /api/auth/whoami both as a session probe (200 means the
JWT is still valid; 401 means we should redirect to the sign-in page) and as
a discovery endpoint for whether real auth is even on (dev-local mode).
"""
from __future__ import annotations

from fastapi import APIRouter

from auth import CurrentUser, auth_mode

router = APIRouter(tags=["auth"])


@router.get("/auth/whoami")
def whoami(user: CurrentUser) -> dict[str, str]:
    return {"user_id": user, "mode": auth_mode()}


@router.get("/auth/mode")
def mode() -> dict[str, str]:
    """Unauthenticated probe — lets the frontend decide whether to render
    the sign-in screen at all (dev-local mode skips it)."""
    return {"mode": auth_mode()}

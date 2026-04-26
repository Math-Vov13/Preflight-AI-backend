"""Supabase JWT auth — single FastAPI dependency wrapping all protected routes.

Two operating modes, decided by whether `settings().supabase_jwt_secret` is set:

1. **Production** — every request must carry `Authorization: Bearer <jwt>`.
   We validate the token's HS256 signature against the Supabase JWT secret,
   then expose `user_id = payload["sub"]` to the route. Anything malformed,
   expired, or signed with the wrong secret returns 401 immediately.

2. **Dev-local** — when the secret is empty (e.g. in a fresh checkout) we
   implicitly authenticate every request as `settings().dev_user_id`
   ("dev-local" by default). This keeps `uv run uvicorn …` runnable on a
   laptop without forcing every contributor through a Supabase signup.
   The frontend sees the same surface either way; in dev-local we just
   skip the sign-in screen.

Usage:

    from app.auth import CurrentUser

    @router.post("/runs/new")
    def start_run(req: ..., user: CurrentUser): ...

`CurrentUser` is a `Annotated[str, Depends(...)]` alias that yields the
user_id directly — routes don't have to know whether auth is on or off.
"""
from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException, status
from jose import JWTError, jwt

from sim_config import settings

logger = logging.getLogger(__name__)

# Supabase signs tokens with `aud="authenticated"` for end-user sessions.
_SUPABASE_AUDIENCE = "authenticated"
_JWT_ALGORITHM = "HS256"


def _bearer_token(authorization: str | None) -> str | None:
    """Pull the raw token from an `Authorization: Bearer <token>` header."""
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def _decode_supabase_jwt(token: str, secret: str) -> dict[str, Any]:
    """Decode + validate a Supabase HS256 JWT, return the full payload.

    Raises HTTPException(401) on signature/audience/exp failures or a
    missing subject claim. Callers that only need the user_id should use
    `_validate_supabase_jwt`; whoami uses this helper to also surface
    exp/iat in the session block.
    """
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[_JWT_ALGORITHM],
            audience=_SUPABASE_AUDIENCE,
        )
    except JWTError as e:
        logger.info("auth: jwt rejected — %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim",
        )
    return payload


def _validate_supabase_jwt(token: str, secret: str) -> str:
    """Back-compat shim — most call sites only need the user_id."""
    return _decode_supabase_jwt(token, secret)["sub"]


def session_payload(authorization: str | None) -> dict[str, Any] | None:
    """Decode the Authorization header and return the JWT payload, or
    None when in dev-local / no header. Public so /auth/whoami can
    surface session expiry without re-implementing the decode flow.
    """
    cfg = settings()
    secret = cfg.supabase_jwt_secret.strip()
    if not secret:
        return None
    token = _bearer_token(authorization)
    if not token:
        return None
    try:
        return _decode_supabase_jwt(token, secret)
    except HTTPException:
        # Already past CurrentUser validation in real callers; if the
        # token races past expiry between deps, treat it as 'no info'.
        return None


def resolve_user(token: str | None) -> str:
    """Map a raw token (or None) to a user_id, raising 401 when production
    mode is active and the token is missing/invalid.

    Shared between the Authorization-header dependency and the SSE query-
    param flow (EventSource can't set custom headers, so /stream takes a
    `?token=<jwt>` query param and routes through here).
    """
    cfg = settings()
    secret = cfg.supabase_jwt_secret.strip()
    if not secret:
        return cfg.dev_user_id
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _validate_supabase_jwt(token, secret)


def get_current_user(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> str:
    """FastAPI dependency: yields the authenticated user_id, raises 401 in
    production mode when the token is missing/invalid."""
    return resolve_user(_bearer_token(authorization))


# Annotated alias so route signatures stay clean — `user: CurrentUser`
# instead of `user: Annotated[str, Depends(get_current_user)]` in every file.
CurrentUser = Annotated[str, Depends(get_current_user)]


def auth_mode() -> str:
    """Returns "supabase" when JWT validation is active, "dev-local" otherwise.
    Used by the /health endpoint + frontend probe so the UI knows whether to
    render the sign-in screen."""
    return "supabase" if settings().supabase_jwt_secret.strip() else "dev-local"

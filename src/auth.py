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
from typing import Annotated

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


def _validate_supabase_jwt(token: str, secret: str) -> str:
    """Decode + validate a Supabase HS256 JWT, return the `sub` claim
    (= the user's auth.uid())."""
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
        # Should not happen for a Supabase-signed token, but guard so the
        # downstream code never sees an empty user_id.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim",
        )
    return sub


def get_current_user(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> str:
    """FastAPI dependency: yields the authenticated user_id, raises 401 in
    production mode when the token is missing/invalid."""
    cfg = settings()
    secret = cfg.supabase_jwt_secret.strip()

    # Dev-local mode — no secret configured, accept all requests under a
    # single shared identity. Logged once at startup, not on every call.
    if not secret:
        return cfg.dev_user_id

    token = _bearer_token(authorization)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _validate_supabase_jwt(token, secret)


# Annotated alias so route signatures stay clean — `user: CurrentUser`
# instead of `user: Annotated[str, Depends(get_current_user)]` in every file.
CurrentUser = Annotated[str, Depends(get_current_user)]


def auth_mode() -> str:
    """Returns "supabase" when JWT validation is active, "dev-local" otherwise.
    Used by the /health endpoint + frontend probe so the UI knows whether to
    render the sign-in screen."""
    return "supabase" if settings().supabase_jwt_secret.strip() else "dev-local"

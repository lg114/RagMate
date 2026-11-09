"""Instance password authentication and session management.

Supports:
1. Localhost auto-bypass: when API_AUTH_TOKEN is unset, loopback requests
   require no password for chat, documents, and settings.
2. Unified remote password: when API_AUTH_TOKEN is set, all non-public endpoints
   require valid password/session (via Cookie, Bearer header, or X-API-Key).
3. Session cookies + Bearer token support with active session invalidation on logout.
"""
from __future__ import annotations

import hmac
import secrets
import time
from urllib.parse import urlparse
from typing import Dict, Set

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.infrastructure.config import settings

router = APIRouter(prefix="/auth", tags=["auth"])

# Active session store: session_token -> creation_timestamp
_active_sessions: Dict[str, float] = {}
_SESSION_TTL: float = 7 * 86400  # 7 days

_LOCAL_IPS: Set[str] = {"127.0.0.1", "::1", "localhost", "testclient"}
_LOCAL_HOST_NAMES: Set[str] = {"localhost", "127.0.0.1", "::1", "testserver"}


def is_local_request(request: Request) -> bool:
    """Validate whether request originates from localhost, verifying client IP, Host, and Origin."""
    # 1. Check client IP
    client_ip = request.client.host if request.client else ""
    if client_ip not in _LOCAL_IPS:
        return False

    # 2. Check Host header (prevent DNS rebinding / host spoofing)
    host_raw = request.headers.get("host", "").split(":")[0].strip().lower()
    if host_raw and host_raw not in _LOCAL_HOST_NAMES:
        return False

    # 3. Check Origin if present
    origin = request.headers.get("origin")
    if origin:
        try:
            parsed = urlparse(origin).hostname
            if parsed and parsed.lower() not in _LOCAL_HOST_NAMES:
                return False
        except Exception:
            return False

    return True


def cleanup_expired_sessions() -> None:
    now = time.time()
    expired = [tok for tok, ts in _active_sessions.items() if now - ts > _SESSION_TTL]
    for tok in expired:
        _active_sessions.pop(tok, None)


def is_authenticated(request: Request, server_token: str) -> bool:
    """Check credentials via Bearer header, X-API-Key header, or ragmate_session cookie."""
    if not server_token:
        return False

    cleanup_expired_sessions()

    # 1. Bearer header
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        bearer_val = auth_header[7:].strip()
        if hmac.compare_digest(bearer_val, server_token) or bearer_val in _active_sessions:
            return True

    # 2. X-API-Key header
    api_key = request.headers.get("x-api-key", "").strip()
    if api_key and (hmac.compare_digest(api_key, server_token) or api_key in _active_sessions):
        return True

    # 3. Cookie session
    cookie_val = request.cookies.get("ragmate_session", "").strip()
    if cookie_val and cookie_val in _active_sessions:
        return True

    return False


class LoginPayload(BaseModel):
    password: str


@router.get("/status")
async def auth_status(request: Request):
    """Report whether authentication is required and if the current caller is authenticated."""
    token = settings.API_AUTH_TOKEN
    is_local = is_local_request(request)
    auth_required = bool(token) or not is_local

    authenticated = False
    if not auth_required:
        authenticated = True
    elif token and is_authenticated(request, token):
        authenticated = True

    return {
        "auth_required": auth_required,
        "authenticated": authenticated,
        "is_local": is_local,
    }


@router.post("/login")
async def login(request: Request, payload: LoginPayload):
    """Verify instance password and establish a session."""
    token = settings.API_AUTH_TOKEN
    if not token:
        # If no password configured on server, login is trivial or not needed
        return {"ok": True, "token": "", "detail": "No password configured on server"}

    if not hmac.compare_digest(payload.password.strip(), token.strip()):
        return JSONResponse(status_code=401, content={"ok": False, "detail": "Incorrect instance password"})

    cleanup_expired_sessions()
    session_token = secrets.token_urlsafe(32)
    _active_sessions[session_token] = time.time()

    response = JSONResponse({"ok": True, "token": session_token})
    response.set_cookie(
        key="ragmate_session",
        value=session_token,
        httponly=True,
        samesite="lax",
        max_age=int(_SESSION_TTL),
    )
    return response


@router.post("/logout")
async def logout(request: Request):
    """Invalidate session and remove cookie."""
    cookie_val = request.cookies.get("ragmate_session", "")
    if cookie_val:
        _active_sessions.pop(cookie_val, None)

    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        bearer_val = auth_header[7:].strip()
        _active_sessions.pop(bearer_val, None)

    response = JSONResponse({"ok": True})
    response.delete_cookie("ragmate_session")
    return response

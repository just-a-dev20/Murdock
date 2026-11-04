"""Authentication for the web API.

A request is allowed if it comes from one of:

* Home Assistant ingress: under the Supervisor, from the ingress proxy's
  fixed address. Home Assistant has already authenticated an admin.
* ``Authorization: Bearer <token>`` with the API token.
* A session cookie set by ``/api/auth/login``. Cookie requests that change
  state also need the ``X-Murdock-UI`` header, which a page on another
  origin can't send without a CORS preflight.

The token comes from ``API_TOKEN`` or is generated into
``<data_dir>/api_token`` on first start.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Optional

_LOGGER = logging.getLogger("murdock.api.auth")

# Fixed address of the Supervisor's ingress proxy.
INGRESS_PROXY_IP = "172.30.32.2"

TOKEN_FILENAME = "api_token"
SESSION_COOKIE = "murdock_session"
UI_HEADER = "x-murdock-ui"
SESSION_MAX_AGE = 90 * 24 * 3600

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_PUBLIC_PATHS = frozenset({
    "/",
    "/api/health",
    "/api/auth/status",
    "/api/auth/login",
    "/api/auth/logout",
})
_PUBLIC_PREFIXES = ("/static/",)

VIA_INGRESS = "ingress"
VIA_TOKEN = "token"
VIA_SESSION = "session"

MISSING_UI_HEADER = "missing X-Murdock-UI header"


def _under_supervisor() -> bool:
    return bool(os.environ.get("SUPERVISOR_TOKEN"))


def _write_private(path: Path, text: str) -> None:
    """Atomically write ``text`` to ``path`` with mode 0600."""
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, text.encode("utf-8"))
    finally:
        os.close(fd)
    os.replace(tmp, path)


def _new_token() -> str:
    return secrets.token_urlsafe(32)


class ApiAuth:
    """The API token and the checks built on it."""

    def __init__(
        self,
        data_dir: Path,
        env_token: Optional[str] = None,
        under_supervisor: Optional[bool] = None,
    ) -> None:
        self._path = Path(data_dir) / TOKEN_FILENAME
        self.under_supervisor = (
            _under_supervisor() if under_supervisor is None else under_supervisor
        )
        env_token = (env_token or "").strip()
        self.pinned = bool(env_token)
        if self.pinned:
            self._token = env_token
        else:
            self._token = self._load_or_create()

    @classmethod
    def from_settings(cls, settings) -> "ApiAuth":
        return cls(
            data_dir=Path(settings.data_dir),
            env_token=getattr(settings, "api_token", None),
        )

    def _load_or_create(self) -> str:
        try:
            token = self._path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            token = ""
        if token:
            return token
        token = _new_token()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        _write_private(self._path, token + "\n")
        _LOGGER.info(
            "Generated an API token in %s (shown in the web UI under "
            "Settings → Home Assistant → API access)", self._path,
        )
        return token

    @property
    def token(self) -> str:
        return self._token

    @property
    def token_path(self) -> Optional[Path]:
        return None if self.pinned else self._path

    def rotate(self) -> str:
        if self.pinned:
            raise RuntimeError("the API token is pinned by API_TOKEN")
        token = _new_token()
        _write_private(self._path, token + "\n")
        self._token = token
        _LOGGER.warning(
            "API token rotated; clients using the old token need the new one"
        )
        return token

    def token_matches(self, candidate: Optional[str]) -> bool:
        if not candidate:
            return False
        return hmac.compare_digest(
            candidate.strip().encode("utf-8"), self._token.encode("utf-8")
        )

    def session_value(self) -> str:
        """HMAC of the token, so rotating the token ends every session."""
        return hmac.new(
            self._token.encode("utf-8"), b"murdock-ui-session", hashlib.sha256
        ).hexdigest()

    def is_ingress(self, scope) -> bool:
        if not self.under_supervisor:
            return False
        client = scope.get("client")
        return bool(client) and client[0] == INGRESS_PROXY_IP

    def authorize(self, scope) -> tuple[Optional[str], Optional[str]]:
        """Return ``(how, None)`` if allowed, else ``(None, reason)``."""
        if self.is_ingress(scope):
            return VIA_INGRESS, None
        headers = _headers(scope)
        auth = headers.get("authorization", "")
        scheme, _, value = auth.partition(" ")
        if scheme.lower() == "bearer" and value:
            if self.token_matches(value):
                return VIA_TOKEN, None
            return None, "invalid API token"
        session = _cookie(headers, SESSION_COOKIE)
        if session and hmac.compare_digest(
            session.encode("utf-8"), self.session_value().encode("utf-8")
        ):
            if scope.get("method", "GET") in _SAFE_METHODS or headers.get(UI_HEADER):
                return VIA_SESSION, None
            return None, MISSING_UI_HEADER
        return None, "authentication required"


def is_public(path: str) -> bool:
    return path in _PUBLIC_PATHS or path.startswith(_PUBLIC_PREFIXES)


def _headers(scope) -> dict:
    out: dict = {}
    for raw_name, raw_value in scope.get("headers") or []:
        out[raw_name.decode("latin-1").lower()] = raw_value.decode("latin-1")
    return out


def _cookie(headers: dict, name: str) -> Optional[str]:
    raw = headers.get("cookie")
    if not raw:
        return None
    jar = SimpleCookie()
    try:
        jar.load(raw)
    except Exception:
        return None
    morsel = jar.get(name)
    return morsel.value if morsel else None


class AuthMiddleware:
    """Reject non-public requests that ``ApiAuth`` doesn't allow."""

    def __init__(self, app, auth: ApiAuth) -> None:
        self.app = app
        self.auth = auth

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in ("http", "websocket") or is_public(scope.get("path", "")):
            await self.app(scope, receive, send)
            return
        how, why = self.auth.authorize(scope)
        if how is not None:
            scope.setdefault("state", {})["auth"] = how
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        status = 403 if why == MISSING_UI_HEADER else 401
        body = json.dumps({"detail": why}).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ]
        if status == 401:
            headers.append((b"www-authenticate", b'Bearer realm="murdock"'))
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

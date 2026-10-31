"""Who may use the web API.

The API hands out everything Murdock knows — the backup alone carries the
Home Assistant token, the MQTT password and every speech-to-text API key —
and it can rewrite all of it. It used to answer anyone who could reach the
port, and a wildcard CORS header let any web page the user happened to
open read the answers too. Now a request is let in for exactly one of
three reasons:

* **Home Assistant ingress.** Running as an add-on, the panel is reached
  through the Supervisor's ingress proxy, which has already logged in a
  Home Assistant admin (``panel_admin: true``). Those requests arrive from
  the proxy's fixed address, ``172.30.32.2``, which nothing else on the
  add-on network can use. Trusted only under the Supervisor.
* **The API token** as ``Authorization: Bearer <token>`` — the Home
  Assistant integration, scripts, curl. A web page on another origin can
  not attach that header without a CORS preflight, which now fails.
* **A browser session** — after the token was entered once in the web UI,
  an ``HttpOnly``, ``SameSite=Strict`` cookie, needed because ``<audio>``
  elements and downloads can't send headers. A cookie rides along on
  requests the user did not mean to make, so anything that changes state
  additionally needs the ``X-Murdock-UI`` header, which the UI always
  sends and a foreign page can't (again: preflight).

The token is a random 256-bit value generated on first start and kept in
``<data dir>/api_token`` (readable on the host in the docker setups), or
pinned with the ``API_TOKEN`` environment variable. The web UI shows it
under *Settings → API access* so it can be pasted into the integration.

Left open on purpose: the UI shell (``/``, ``/static``) so the login
prompt can load, ``/api/health`` for container health checks, and the
login endpoints themselves.
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

# The Supervisor's ingress proxy. Fixed by Home Assistant; add-ons are
# told to accept ingress traffic from this address only.
INGRESS_PROXY_IP = "172.30.32.2"

TOKEN_FILENAME = "api_token"
SESSION_COOKIE = "murdock_session"
UI_HEADER = "x-murdock-ui"
# A browser that entered the token once stays signed in this long.
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

# Reasons a request was let in, as reported by ``authorize``.
VIA_INGRESS = "ingress"
VIA_TOKEN = "token"
VIA_SESSION = "session"


def _under_supervisor() -> bool:
    return bool(os.environ.get("SUPERVISOR_TOKEN"))


def _write_private(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` readable by the owner only, atomically."""
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
        # Pinned by the environment: the file is neither read nor written,
        # and the token can't be rotated from the UI.
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
            "Generated an API token in %s — the web UI shows it under "
            "Settings → API access", self._path,
        )
        return token

    @property
    def token(self) -> str:
        return self._token

    @property
    def token_path(self) -> Optional[Path]:
        return None if self.pinned else self._path

    def rotate(self) -> str:
        """Replace the token. Every client using the old one is shut out."""
        if self.pinned:
            raise RuntimeError("the API token is pinned by API_TOKEN")
        token = _new_token()
        _write_private(self._path, token + "\n")
        self._token = token
        _LOGGER.warning(
            "API token rotated — the Home Assistant integration and other "
            "clients need the new one"
        )
        return token

    def token_matches(self, candidate: Optional[str]) -> bool:
        if not candidate:
            return False
        return hmac.compare_digest(
            candidate.strip().encode("utf-8"), self._token.encode("utf-8")
        )

    def session_value(self) -> str:
        """Cookie value for a signed-in browser.

        Derived from the token rather than the token itself, so the cookie
        is not a credential for the Bearer header, and rotating the token
        signs every browser out.
        """
        return hmac.new(
            self._token.encode("utf-8"), b"murdock-ui-session", hashlib.sha256
        ).hexdigest()

    # ------------------------------------------------------------------
    # Request checks (on raw ASGI scopes, so the middleware stays cheap)
    # ------------------------------------------------------------------

    def is_ingress(self, scope) -> bool:
        if not self.under_supervisor:
            return False
        client = scope.get("client")
        return bool(client) and client[0] == INGRESS_PROXY_IP

    def authorize(self, scope) -> tuple[Optional[str], Optional[str]]:
        """``(how, None)`` when the request may proceed, else ``(None, why)``."""
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
            return None, "missing X-Murdock-UI header"
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
    """Pure ASGI: refuses every non-public request ``ApiAuth`` doesn't allow."""

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
        status = 403 if why == "missing X-Murdock-UI header" else 401
        body = json.dumps({"detail": why}).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ]
        if status == 401:
            headers.append((b"www-authenticate", b'Bearer realm="murdock"'))
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

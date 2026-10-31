"""Sign-in for the web UI and management of the API token (see auth.py)."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from .auth import SESSION_COOKIE, SESSION_MAX_AGE, ApiAuth

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _auth(request: Request) -> ApiAuth:
    return request.app.state.auth


def _set_session(request: Request, response: Response, auth: ApiAuth) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        auth.session_value(),
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
        path="/",
    )


class AuthStatus(BaseModel):
    authenticated: bool
    # "ingress", "token" or "session" when authenticated.
    via: Optional[str] = None


class LoginIn(BaseModel):
    token: str


class TokenOut(BaseModel):
    token: str
    # Set by the API_TOKEN environment variable: can't be rotated here.
    pinned: bool
    # Where the token is kept on disk, when it isn't pinned.
    path: Optional[str] = None


@router.get("/status", response_model=AuthStatus)
async def auth_status(request: Request) -> AuthStatus:
    """Whether this browser is signed in — public, so the UI can ask."""
    how, _why = _auth(request).authorize(request.scope)
    return AuthStatus(authenticated=how is not None, via=how)


@router.post("/login", response_model=AuthStatus)
async def login(body: LoginIn, request: Request, response: Response) -> AuthStatus:
    """Exchange the API token for a browser session cookie."""
    auth = _auth(request)
    if not auth.token_matches(body.token):
        raise HTTPException(status_code=401, detail="invalid API token")
    _set_session(request, response, auth)
    return AuthStatus(authenticated=True, via="session")


@router.post("/logout", response_model=AuthStatus)
async def logout(response: Response) -> AuthStatus:
    response.delete_cookie(SESSION_COOKIE, path="/")
    return AuthStatus(authenticated=False)


def _token_out(auth: ApiAuth) -> TokenOut:
    path = auth.token_path
    return TokenOut(token=auth.token, pinned=auth.pinned, path=str(path) if path else None)


@router.get("/token", response_model=TokenOut)
async def get_token(request: Request) -> TokenOut:
    """The API token, for pasting into the Home Assistant integration.

    Behind the auth middleware like everything else: only somebody who is
    already in (ingress admin, token, or signed-in browser) can read it.
    """
    return _token_out(_auth(request))


@router.post("/token/rotate", response_model=TokenOut)
async def rotate_token(request: Request, response: Response) -> TokenOut:
    """Replace the token, e.g. after it leaked.

    The browser that asked stays signed in; every other client, the Home
    Assistant integration included, needs the new token.
    """
    auth = _auth(request)
    if auth.pinned:
        raise HTTPException(
            status_code=409,
            detail="The token is set by the API_TOKEN environment variable — change it there.",
        )
    auth.rotate()
    if getattr(request.state, "auth", None) == "session":
        _set_session(request, response, auth)
    return _token_out(auth)

"""Web UI sign-in and API token management."""

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
    via: Optional[str] = None


class LoginIn(BaseModel):
    token: str


class TokenOut(BaseModel):
    token: str
    pinned: bool
    path: Optional[str] = None


@router.get("/status", response_model=AuthStatus)
async def auth_status(request: Request) -> AuthStatus:
    """Whether the request is authenticated."""
    how, _why = _auth(request).authorize(request.scope)
    return AuthStatus(authenticated=how is not None, via=how)


@router.post("/login", response_model=AuthStatus)
async def login(body: LoginIn, request: Request, response: Response) -> AuthStatus:
    """Exchange the API token for a session cookie."""
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
    """Return the API token."""
    return _token_out(_auth(request))


@router.post("/token/rotate", response_model=TokenOut)
async def rotate_token(request: Request, response: Response) -> TokenOut:
    """Replace the API token. The calling browser stays signed in."""
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

"""Web API authentication: ingress, API token, or browser session."""

from __future__ import annotations

import io
import stat
import zipfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from murdock.api.app import create_app
from murdock.api.auth import INGRESS_PROXY_IP, SESSION_COOKIE, ApiAuth
from murdock.config import Settings
from murdock.core.context import AppContext
from murdock.core.db import open_db, set_setting

SECRETS = ("HA-LONG-LIVED-TOKEN", "MQTT-PASSWORD", "sk-OPENAI-KEY")
EVIL = "https://evil.example"


def _context(tmp_path):
    db = open_db(tmp_path / "m.db")
    set_setting(db, "ha_token", SECRETS[0])
    set_setting(db, "mqtt_password", SECRETS[1])
    ctx = AppContext(
        settings=Settings(data_dir=tmp_path), db=db, embedder=None, vad=None,
        speakers=SimpleNamespace(list_speakers=lambda: []),
        unknown=None, ha=None, mqtt=None, recognition=None,
    )
    ctx.stt_services.create("OpenAI", "openai", {"api_key": SECRETS[2], "model": "m"})
    return ctx


def _client(tmp_path, *, supervisor=False, address="testclient", env_token=None):
    auth = ApiAuth(tmp_path, env_token=env_token, under_supervisor=supervisor)
    app = create_app(_context(tmp_path), auth=auth)
    return TestClient(app, client=(address, 50000)), auth


def _secrets_in(response) -> list[str]:
    z = zipfile.ZipFile(io.BytesIO(response.content))
    blob = "".join(z.read(n).decode("utf-8", "replace") for n in z.namelist())
    return [s for s in SECRETS if s in blob]


# --- Anonymous and cross-origin access -----------------------------------


def test_an_anonymous_page_cannot_read_the_backup(tmp_path):
    client, _ = _client(tmp_path)
    r = client.get("/api/backup", headers={"Origin": EVIL})
    assert r.status_code == 401
    assert SECRETS[0] not in r.text
    assert "access-control-allow-origin" not in r.headers


def test_no_cors_grant_for_any_origin(tmp_path):
    client, auth = _client(tmp_path)
    r = client.get(
        "/api/backup",
        headers={"Origin": EVIL, "Authorization": f"Bearer {auth.token}"},
    )
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers
    pre = client.options(
        "/api/backup/restore",
        headers={"Origin": EVIL, "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in pre.headers


@pytest.mark.parametrize("path", [
    "/api/backup", "/api/settings", "/api/speakers", "/api/stt-services",
    "/api/version", "/api/state", "/api/auth/token", "/docs", "/openapi.json",
])
def test_everything_but_the_public_list_needs_auth(tmp_path, path):
    client, _ = _client(tmp_path)
    assert client.get(path).status_code == 401


def test_writes_need_auth_too(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post(
        "/api/backup/restore?mode=replace",
        files={"file": ("b.zip", b"PK\x05\x06" + b"\x00" * 18, "application/zip")},
    )
    assert r.status_code == 401


def test_public_paths_stay_open(tmp_path):
    client, _ = _client(tmp_path)
    assert client.get("/api/health").status_code == 200   # container health checks
    assert client.get("/").status_code == 200             # the UI shell / login prompt
    assert client.get("/static/app.js").status_code == 200
    status = client.get("/api/auth/status").json()
    assert status == {"authenticated": False, "via": None}


# --- The API token --------------------------------------------------------


def test_the_token_opens_the_api(tmp_path):
    client, auth = _client(tmp_path)
    r = client.get("/api/backup", headers={"Authorization": f"Bearer {auth.token}"})
    assert r.status_code == 200
    assert _secrets_in(r) == list(SECRETS)  # the backup itself is unchanged


def test_a_wrong_token_is_refused(tmp_path):
    client, auth = _client(tmp_path)
    r = client.get("/api/backup", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401
    assert r.headers["www-authenticate"].startswith("Bearer")


def test_the_token_is_generated_once_and_kept_private(tmp_path):
    first = ApiAuth(tmp_path, under_supervisor=False)
    path = tmp_path / "api_token"
    assert path.read_text().strip() == first.token
    assert len(first.token) >= 40
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert ApiAuth(tmp_path, under_supervisor=False).token == first.token


def test_an_environment_token_is_pinned_and_never_written(tmp_path):
    auth = ApiAuth(tmp_path, env_token="  pinned-token  ", under_supervisor=False)
    assert auth.token == "pinned-token" and auth.pinned
    assert not (tmp_path / "api_token").exists()
    with pytest.raises(RuntimeError):
        auth.rotate()


def test_the_api_token_setting_comes_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("API_TOKEN", "from-env")
    auth = ApiAuth.from_settings(Settings(data_dir=tmp_path))
    assert auth.token == "from-env" and auth.pinned


# --- Home Assistant ingress ----------------------------------------------


def test_ingress_is_trusted_under_the_supervisor(tmp_path):
    client, _ = _client(tmp_path, supervisor=True, address=INGRESS_PROXY_IP)
    assert client.get("/api/backup").status_code == 200
    assert client.get("/api/auth/status").json() == {"authenticated": True, "via": "ingress"}


def test_the_ingress_address_means_nothing_outside_the_supervisor(tmp_path):
    client, _ = _client(tmp_path, supervisor=False, address=INGRESS_PROXY_IP)
    assert client.get("/api/backup").status_code == 401


def test_other_addresses_are_not_ingress(tmp_path):
    # The add-on network gateway: HA Core and port-forwarded LAN traffic.
    client, _ = _client(tmp_path, supervisor=True, address="172.30.32.1")
    assert client.get("/api/backup").status_code == 401


def test_a_forged_ingress_header_or_forwarded_address_does_not_help(tmp_path):
    client, _ = _client(tmp_path, supervisor=True)
    r = client.get("/api/backup", headers={
        "X-Ingress-Path": "/api/hassio_ingress/abc",
        "X-Forwarded-For": INGRESS_PROXY_IP,
    })
    assert r.status_code == 401


# --- Browser session ------------------------------------------------------


def test_login_with_the_wrong_token_fails(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post("/api/auth/login", json={"token": "guess"})
    assert r.status_code == 401
    assert SESSION_COOKIE not in r.cookies


def test_login_sets_a_strict_httponly_session(tmp_path):
    client, auth = _client(tmp_path)
    r = client.post("/api/auth/login", json={"token": auth.token})
    assert r.status_code == 200
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{SESSION_COOKIE}=")
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie
    assert auth.token not in cookie
    assert client.get("/api/backup").status_code == 200
    assert client.get("/api/auth/status").json()["via"] == "session"


def test_a_cookie_alone_cannot_change_anything(tmp_path):
    client, auth = _client(tmp_path)
    client.post("/api/auth/login", json={"token": auth.token})
    r = client.post("/api/auth/token/rotate")
    assert r.status_code == 403
    r = client.post("/api/auth/token/rotate", headers={"X-Murdock-UI": "1"})
    assert r.status_code == 200


def test_logout_ends_the_session(tmp_path):
    client, auth = _client(tmp_path)
    client.post("/api/auth/login", json={"token": auth.token})
    client.post("/api/auth/logout")
    assert client.get("/api/backup").status_code == 401


# --- Showing and rotating the token ---------------------------------------


def test_the_ui_can_show_the_token_for_the_integration(tmp_path):
    client, auth = _client(tmp_path, supervisor=True, address=INGRESS_PROXY_IP)
    out = client.get("/api/auth/token").json()
    assert out["token"] == auth.token and out["pinned"] is False
    assert out["path"].endswith("api_token")


def test_rotation_shuts_out_the_old_token_but_keeps_the_browser(tmp_path):
    client, auth = _client(tmp_path)
    old = auth.token
    client.post("/api/auth/login", json={"token": old})
    r = client.post("/api/auth/token/rotate", headers={"X-Murdock-UI": "1"})
    new = r.json()["token"]
    assert new != old
    assert (tmp_path / "api_token").read_text().strip() == new
    assert client.get("/api/backup").status_code == 200  # fresh cookie
    fresh, _ = _client(tmp_path)
    assert fresh.get("/api/backup", headers={"Authorization": f"Bearer {old}"}).status_code == 401
    assert fresh.get("/api/backup", headers={"Authorization": f"Bearer {new}"}).status_code == 200


def test_a_pinned_token_cannot_be_rotated_from_the_ui(tmp_path):
    client, auth = _client(tmp_path, env_token="pinned")
    r = client.post("/api/auth/token/rotate", headers={"Authorization": "Bearer pinned"})
    assert r.status_code == 409
    assert auth.token == "pinned"

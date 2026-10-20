from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import connect
from app.main import create_app


def _app(settings, kb_services, password="adminpass123", username="admin"):
    return create_app(
        replace(settings, admin_username=username, admin_password=password), kb_services
    )


def _login(app, username="admin", password="adminpass123"):
    c = TestClient(app)
    return c, c.post("/api/auth/login", json={"username": username, "password": password})


def _hash(settings, username="admin"):
    conn = connect(settings.db_path)
    try:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
        return row["password_hash"] if row else None
    finally:
        conn.close()


def test_admin_account_is_created_at_startup(settings, kb_services):
    c, r = _login(_app(settings, kb_services))
    assert r.status_code == 200
    assert c.get("/api/me").json()["is_admin"] is True


def test_nobody_else_is_admin(settings, kb_services):
    c = TestClient(_app(settings, kb_services))
    creds = {"username": "bob", "password": "password123"}
    assert c.post("/api/auth/register", json=creds).status_code == 201
    assert c.post("/api/auth/login", json=creds).status_code == 200
    assert c.get("/api/me").json()["is_admin"] is False


def test_admin_username_cannot_be_registered(settings, kb_services):
    c = TestClient(_app(settings, kb_services))
    r = c.post("/api/auth/register", json={"username": "admin", "password": "otherpass123"})
    assert r.status_code == 409


def test_changed_password_in_env_wins_on_restart(settings, kb_services):
    old_client, r = _login(_app(settings, kb_services))
    assert r.status_code == 200

    app = _app(settings, kb_services, password="newpass12345")
    assert _login(app)[1].status_code == 401
    assert _login(app, password="newpass12345")[1].status_code == 200
    # sessions opened with the old password are signed out
    assert old_client.get("/api/me").status_code == 401


def test_unchanged_password_keeps_hash_and_sessions(settings, kb_services):
    c, _ = _login(_app(settings, kb_services))
    before = _hash(settings)
    _app(settings, kb_services)
    assert _hash(settings) == before
    assert c.get("/api/me").status_code == 200


def test_existing_user_with_admin_name_takes_env_password(settings, kb_services):
    plain = TestClient(create_app(settings, kb_services))
    creds = {"username": "admin", "password": "squatter123"}
    assert plain.post("/api/auth/register", json=creds).status_code == 201

    app = _app(settings, kb_services)
    assert _login(app, password="squatter123")[1].status_code == 401
    c, r = _login(app)
    assert r.status_code == 200 and c.get("/api/me").json()["is_admin"] is True


def test_no_admin_configured_means_no_admin(settings, kb_services):
    c = TestClient(create_app(settings, kb_services))
    creds = {"username": "admin", "password": "password123"}
    assert c.post("/api/auth/register", json=creds).status_code == 201
    assert c.post("/api/auth/login", json=creds).status_code == 200
    assert c.get("/api/me").json()["is_admin"] is False


def test_admin_env_parsing(monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", " admin ")
    monkeypatch.setenv("ADMIN_PASSWORD", "adminpass123")
    s = Settings.from_env()
    assert (s.admin_username, s.admin_password) == ("admin", "adminpass123")
    assert "adminpass123" not in repr(s)


def test_admin_env_unset(monkeypatch):
    monkeypatch.delenv("ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    s = Settings.from_env()
    assert (s.admin_username, s.admin_password) == ("", "")


def test_old_admin_usernames_setting_is_rejected(monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAMES", "alice")
    with pytest.raises(ValueError, match="ADMIN_USERNAMES 已不再使用"):
        Settings.from_env()
    monkeypatch.setenv("ADMIN_USERNAMES", "")
    Settings.from_env()


@pytest.mark.parametrize(
    ("username", "password", "message"),
    [
        ("admin", "", "ADMIN_USERNAME 與 ADMIN_PASSWORD 必須同時設定"),
        ("", "adminpass123", "ADMIN_USERNAME 與 ADMIN_PASSWORD 必須同時設定"),
        ("ab", "adminpass123", "ADMIN_USERNAME 長度需為 3 到 32 個字元"),
        ("admin", "short", "ADMIN_PASSWORD 至少需要 8 個字元"),
        ("admin", "x" * 73, "ADMIN_PASSWORD 最多 72 個位元組"),
    ],
)
def test_invalid_admin_env_fails_fast(monkeypatch, username, password, message):
    monkeypatch.setenv("ADMIN_USERNAME", username)
    monkeypatch.setenv("ADMIN_PASSWORD", password)
    with pytest.raises(ValueError, match=message):
        Settings.from_env()

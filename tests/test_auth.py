from app.db import connect

CREDS = {"username": "alice", "password": "password123"}


def test_register_returns_201(client):
    r = client.post("/api/auth/register", json=CREDS)
    assert r.status_code == 201 and r.json()["username"] == "alice"


def test_register_duplicate_username_409(client):
    client.post("/api/auth/register", json=CREDS)
    assert client.post("/api/auth/register", json=CREDS).status_code == 409


def test_register_rejects_short_username_and_password(client):
    r = client.post("/api/auth/register", json={"username": "ab", "password": "password123"})
    assert r.status_code == 422
    r = client.post("/api/auth/register", json={"username": "alice", "password": "short"})
    assert r.status_code == 422


def test_register_rejects_username_over_32_chars(client):
    r = client.post("/api/auth/register", json={"username": "a" * 33, "password": "password123"})
    assert r.status_code == 422


def test_register_rejects_password_over_72_bytes(client):
    r = client.post("/api/auth/register", json={"username": "bob", "password": "密" * 25})
    assert r.status_code == 422


def test_password_is_hashed_in_db(client, settings):
    client.post("/api/auth/register", json=CREDS)
    row = connect(settings.db_path).execute("SELECT password_hash FROM users").fetchone()
    assert row["password_hash"] != "password123"
    assert row["password_hash"].startswith("$2")


def test_login_sets_httponly_cookie_and_me_works(client):
    client.post("/api/auth/register", json=CREDS)
    r = client.post("/api/auth/login", json=CREDS)
    assert r.status_code == 200
    cookie = r.headers["set-cookie"]
    assert "session=" in cookie and "HttpOnly" in cookie
    assert "samesite=lax" in cookie.lower()
    assert "max-age=604800" in cookie.lower()
    me = client.get("/api/me")
    assert me.status_code == 200 and me.json()["username"] == "alice"


def test_login_wrong_password_401(client):
    client.post("/api/auth/register", json=CREDS)
    r = client.post("/api/auth/login", json={**CREDS, "password": "wrongpass1"})
    assert r.status_code == 401


def test_login_unknown_user_401_same_detail_as_wrong_password(client):
    client.post("/api/auth/register", json=CREDS)
    wrong = client.post("/api/auth/login", json={**CREDS, "password": "wrongpass1"})
    unknown = client.post("/api/auth/login", json={"username": "nobody", "password": "password123"})
    assert unknown.status_code == 401
    assert unknown.json()["detail"] == wrong.json()["detail"]


def test_me_without_cookie_401(client):
    assert client.get("/api/me").status_code == 401


def test_logout_invalidates_session(make_user):
    c = make_user("alice")
    token = c.cookies.get("session")
    assert c.post("/api/auth/logout").status_code == 204
    c.cookies.set("session", token)
    assert c.get("/api/me").status_code == 401


def test_expired_session_401(make_user, settings):
    c = make_user("alice")
    conn = connect(settings.db_path)
    conn.execute("UPDATE sessions SET expires_at = '2000-01-01T00:00:00+00:00'")
    conn.commit()
    assert c.get("/api/me").status_code == 401

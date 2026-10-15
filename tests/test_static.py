from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_login_page_served(client):
    assert client.get("/login.html").status_code == 200


def test_api_routes_not_shadowed(client):
    assert client.get("/api/me").status_code == 401


def test_no_innerhtml_in_scripts():
    for f in ["static/app.js", "static/login.js"]:
        assert "innerHTML" not in (ROOT / f).read_text(encoding="utf-8")

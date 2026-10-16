from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

PLACEHOLDER = "前端尚未建置，請在 frontend/ 執行 npm run build"


def _app_with_dist(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>phoebe</html>", encoding="utf-8")
    return create_app(
        Settings(db_path=str(tmp_path / "t.db"), frontend_dist=str(dist))
    )


def test_serves_built_frontend(tmp_path):
    r = TestClient(_app_with_dist(tmp_path)).get("/")
    assert r.status_code == 200
    assert "phoebe" in r.text


def test_placeholder_when_frontend_not_built(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.text == PLACEHOLDER


def test_api_routes_not_shadowed_by_frontend(tmp_path):
    assert TestClient(_app_with_dist(tmp_path)).get("/api/me").status_code == 401


def test_settings_frontend_dist_default_and_env(monkeypatch):
    monkeypatch.delenv("FRONTEND_DIST", raising=False)
    p = Path(Settings.from_env().frontend_dist)
    assert p.is_absolute()
    assert p.parts[-2:] == ("frontend", "dist")
    monkeypatch.setenv("FRONTEND_DIST", "/x/y")
    assert Settings.from_env().frontend_dist == "/x/y"

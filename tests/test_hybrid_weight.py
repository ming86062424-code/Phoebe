"""The vector/keyword weight of the hybrid search: settings, search API and chat."""

import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.routes import chat as chat_routes
from tests.fakes import FakeChatClient


def _login(app, username="alice"):
    c = TestClient(app)
    creds = {"username": username, "password": "password123"}
    assert c.post("/api/auth/register", json=creds).status_code == 201
    assert c.post("/api/auth/login", json=creds).status_code == 200
    return c


@pytest.fixture
def weights(kb_services, monkeypatch):
    """Records the vector_weight of every search."""
    seen: list[float] = []
    real = kb_services.index.search

    def search(kb_id, query, top_k, vector_weight=0.5):
        seen.append(vector_weight)
        return real(kb_id, query, top_k, vector_weight)

    monkeypatch.setattr(kb_services.index, "search", search)
    return seen


def _app(settings, kb_services, weight=0.3):
    return create_app(replace(settings, kb_vector_weight=weight), kb_services)


def _kb(c):
    return c.post("/api/kbs", json={"name": "k"}).json()["id"]


def test_search_uses_the_configured_weight_by_default(settings, kb_services, weights):
    c = _login(_app(settings, kb_services))
    assert c.get(f"/api/kbs/{_kb(c)}/search", params={"q": "x"}).status_code == 200
    assert weights == [0.3]


def test_search_takes_a_weight_for_testing(settings, kb_services, weights):
    c = _login(_app(settings, kb_services))
    kb = _kb(c)
    assert c.get(f"/api/kbs/{kb}/search", params={"q": "x", "vector_weight": 0.8}).status_code == 200
    assert c.get(f"/api/kbs/{kb}/search", params={"q": "x", "vector_weight": 0}).status_code == 200
    assert weights == [0.8, 0.0]


@pytest.mark.parametrize("weight", ["-0.1", "1.01", "abc"])
def test_search_rejects_a_weight_outside_zero_to_one(settings, kb_services, weight):
    c = _login(_app(settings, kb_services))
    r = c.get(f"/api/kbs/{_kb(c)}/search", params={"q": "x", "vector_weight": weight})
    assert r.status_code == 422


def test_search_settings_endpoint(settings, kb_services):
    app = _app(settings, kb_services)
    assert TestClient(app).get("/api/search-settings").status_code == 401
    c = _login(app)
    assert c.get("/api/search-settings").json() == {"vector_weight": 0.3, "top_k": settings.kb_top_k}


def test_chat_uses_the_configured_weight(settings, kb_services, weights, monkeypatch):
    app = _app(settings, kb_services, weight=0.7)
    fake = FakeChatClient()
    app.dependency_overrides[chat_routes.get_chat_client] = lambda: fake
    c = _login(app)
    conv = c.post("/api/conversations").json()
    c.patch(f"/api/conversations/{conv['id']}", json={"kb_id": _kb(c)})
    r = c.post(f"/api/conversations/{conv['id']}/messages", json={"content": "嗨"})
    assert r.status_code == 200
    assert [json.loads(line)["type"] for line in r.text.splitlines()][-1] == "done"
    assert weights == [0.7]


def test_kb_vector_weight_env(monkeypatch):
    monkeypatch.delenv("KB_VECTOR_WEIGHT", raising=False)
    assert Settings.from_env().kb_vector_weight == 0.5
    monkeypatch.setenv("KB_VECTOR_WEIGHT", "0.3")
    assert Settings.from_env().kb_vector_weight == 0.3


@pytest.mark.parametrize("value", ["-0.1", "1.5", "half"])
def test_kb_vector_weight_env_must_be_zero_to_one(monkeypatch, value):
    monkeypatch.setenv("KB_VECTOR_WEIGHT", value)
    with pytest.raises(ValueError, match="KB_VECTOR_WEIGHT 必須是 0 到 1 之間的數字"):
        Settings.from_env()

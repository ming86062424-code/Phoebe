from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.llm.ollama import OllamaChatClient
from app.llm.gemini import GeminiChatClient
from app.main import create_app

_REAL_AVAILABILITY = OllamaChatClient.availability


@pytest.fixture(autouse=True)
def fake_ollama(monkeypatch):
    async def availability(self):
        return True, ""

    async def list_models(self):
        return ["m1", "m2"]

    monkeypatch.setattr(OllamaChatClient, "availability", availability)
    monkeypatch.setattr(OllamaChatClient, "list_models", list_models)


def _client(settings, kb_services, username, **overrides):
    app = create_app(replace(settings, admin_usernames=("admin",), **overrides), kb_services)
    c = TestClient(app)
    creds = {"username": username, "password": "password123"}
    assert c.post("/api/auth/register", json=creds).status_code == 201
    assert c.post("/api/auth/login", json=creds).status_code == 200
    return c


@pytest.fixture
def admin(settings, kb_services):
    return _client(settings, kb_services, "admin")


def test_me_reports_is_admin(settings, kb_services):
    assert _client(settings, kb_services, "admin").get("/api/me").json()["is_admin"] is True
    assert _client(settings, kb_services, "bob").get("/api/me").json()["is_admin"] is False


def test_non_admin_gets_403(settings, kb_services):
    c = _client(settings, kb_services, "bob")
    for r in (
        c.get("/api/settings"),
        c.put("/api/settings", json={"system_prompt": "x"}),
        c.get("/api/settings/models?provider=ollama"),
    ):
        assert r.status_code == 403 and r.json()["detail"] == "需要管理員權限"


def test_anonymous_gets_401(client):
    assert client.get("/api/settings").status_code == 401


def test_get_settings_reports_provider_availability(admin):
    body = admin.get("/api/settings").json()
    assert body["llm_provider"] == "ollama"
    assert body["temperature"] is None
    assert body["providers"]["ollama"] == {"available": True, "reason": ""}
    assert body["providers"]["gemini"] == {
        "available": False,
        "reason": "未設定 GEMINI_API_KEY",
    }


def test_put_updates_partially(admin):
    r = admin.put("/api/settings", json={"system_prompt": "hi", "temperature": 0.7})
    assert r.status_code == 200 and r.json()["system_prompt"] == "hi"
    r = admin.put("/api/settings", json={"ollama_model": "m2"})
    body = admin.get("/api/settings").json()
    assert (body["system_prompt"], body["temperature"], body["ollama_model"]) == ("hi", 0.7, "m2")
    admin.put("/api/settings", json={"temperature": None})
    assert admin.get("/api/settings").json()["temperature"] is None


@pytest.mark.parametrize(
    "body",
    [
        {"llm_provider": "x"},
        {"temperature": 2.5},
        {"system_prompt": "a" * 4001},
        {"ollama_model": ""},
        {"unknown": 1},
        {"llm_provider": None},
    ],
)
def test_put_rejects_bad_values_422(admin, body):
    assert admin.put("/api/settings", json=body).status_code == 422


def test_put_to_unavailable_provider_is_409(admin):
    r = admin.put("/api/settings", json={"llm_provider": "gemini"})
    assert r.status_code == 409
    assert r.json()["detail"] == "Gemini 目前不可用：未設定 GEMINI_API_KEY"
    assert admin.get("/api/settings").json()["llm_provider"] == "ollama"


def test_put_to_gemini_with_key_works(settings, kb_services):
    c = _client(settings, kb_services, "admin", gemini_api_key="sk-secret")
    r = c.put("/api/settings", json={"llm_provider": "gemini"})
    assert r.status_code == 200 and r.json()["llm_provider"] == "gemini"


def test_switching_to_unavailable_ollama_is_409(settings, kb_services, monkeypatch):
    admin = _client(settings, kb_services, "admin", gemini_api_key="k")
    assert admin.put("/api/settings", json={"llm_provider": "gemini"}).status_code == 200

    async def down(self):
        return False, "無法連線到 Ollama"

    monkeypatch.setattr(OllamaChatClient, "availability", down)
    r = admin.put("/api/settings", json={"llm_provider": "ollama"})
    assert r.status_code == 409
    assert r.json()["detail"] == "Ollama 目前不可用：無法連線到 Ollama"


def test_models_endpoint(admin, settings, kb_services):
    assert admin.get("/api/settings/models?provider=ollama").json() == {"models": ["m1", "m2"]}
    assert admin.get("/api/settings/models?provider=x").status_code == 422
    assert admin.get("/api/settings/models").status_code == 422
    r = admin.get("/api/settings/models?provider=gemini")
    assert r.status_code == 503 and r.json()["detail"] == "未設定 GEMINI_API_KEY"


def test_models_ollama_unreachable_is_503(admin, monkeypatch):
    from app.llm import LlmUnavailable

    async def boom(self):
        raise LlmUnavailable()

    monkeypatch.setattr(OllamaChatClient, "list_models", boom)
    r = admin.get("/api/settings/models?provider=ollama")
    assert r.status_code == 503 and r.json()["detail"] == "無法連線到 Ollama"


def test_settings_response_never_contains_api_key(settings, kb_services, monkeypatch):
    async def fake_models(self):
        return ["gemini-2.5-flash"]

    monkeypatch.setattr(GeminiChatClient, "list_models", fake_models)
    c = _client(settings, kb_services, "admin", gemini_api_key="sk-secret")
    c.put("/api/settings", json={"llm_provider": "gemini"})
    for r in (
        c.get("/api/settings"),
        c.get("/api/settings/models?provider=gemini"),
        c.put("/api/settings", json={"system_prompt": "x"}),
        c.put("/api/settings", json={"temperature": 9}),
    ):
        assert "sk-secret" not in r.text


def test_saving_other_fields_while_current_provider_down_is_ok(admin, monkeypatch):
    async def down(self):
        return False, "無法連線到 Ollama"

    monkeypatch.setattr(OllamaChatClient, "availability", down)
    r = admin.put("/api/settings", json={"llm_provider": "ollama", "system_prompt": "x"})
    assert r.status_code == 200 and r.json()["system_prompt"] == "x"
    r = admin.put("/api/settings", json={"llm_provider": "gemini"})
    assert r.status_code == 409


@pytest.mark.anyio
async def test_ollama_availability_uses_short_timeout():
    import httpx

    seen = []

    def handler(request):
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json={"models": []})

    client = OllamaChatClient("http://x", "m", transport=httpx.MockTransport(handler))
    assert await _REAL_AVAILABILITY(client) == (True, "")
    assert max(v for v in seen[0].values() if v) <= 3.0


def test_gemini_models_endpoint_lists_and_maps_errors(settings, kb_services, monkeypatch):
    from app.llm import LlmAuthError

    c = _client(settings, kb_services, "admin", gemini_api_key="sk-secret")

    async def ok(self):
        return ["gemini-2.5-flash"]

    monkeypatch.setattr(GeminiChatClient, "list_models", ok)
    assert c.get("/api/settings/models?provider=gemini").json() == {
        "models": ["gemini-2.5-flash"]
    }

    async def bad(self):
        raise LlmAuthError("Gemini API key 無效")

    monkeypatch.setattr(GeminiChatClient, "list_models", bad)
    r = c.get("/api/settings/models?provider=gemini")
    assert r.status_code == 503 and r.json()["detail"] == "Gemini API key 無效"

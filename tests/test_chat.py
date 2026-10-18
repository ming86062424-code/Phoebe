import json

from app.deps import get_chat_client
from app.ollama_client import ChatDelta, ModelNotFound, OllamaStreamError, OllamaUnavailable
from tests.fakes import FakeChatClient


def _new_conv(c):
    return c.post("/api/conversations").json()["id"]


def _send(c, cid, content):
    return c.post(f"/api/conversations/{cid}/messages", json={"content": content})


def _messages(c, cid):
    return c.get(f"/api/conversations/{cid}").json()["messages"]


def events(r):
    return [json.loads(line) for line in r.text.splitlines() if line]


def test_stream_emits_thinking_content_done(make_user, fake_chat):
    c = make_user("alice")
    cid = _new_conv(c)
    r = _send(c, cid, "嗨")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    ev = events(r)
    assert ev[:3] == [
        {"type": "thinking", "text": "嗯"},
        {"type": "content", "text": "你"},
        {"type": "content", "text": "好"},
    ]
    assert ev[-1]["type"] == "done"
    assistant = _messages(c, cid)[1]
    assert assistant["id"] == ev[-1]["message_id"]
    assert assistant["content"] == "你好"


def test_history_and_db_exclude_thinking(make_user, fake_chat):
    c = make_user("alice")
    cid = _new_conv(c)
    _send(c, cid, "第一句")
    _send(c, cid, "第二句")
    assert fake_chat.calls[1]["messages"][1] == {"role": "assistant", "content": "你好"}
    assert _messages(c, cid)[1]["content"] == "你好"


def test_ndjson_keeps_chinese_unescaped(make_user, fake_chat):
    c = make_user("alice")
    cid = _new_conv(c)
    r = _send(c, cid, "嗨")
    assert "\\u" not in r.text
    assert "你" in r.text


def test_thinking_only_reply_saves_empty_assistant(make_user, app):
    app.dependency_overrides[get_chat_client] = lambda: FakeChatClient(
        deltas=[ChatDelta("thinking", "嗯")]
    )
    c = make_user("alice")
    cid = _new_conv(c)
    ev = events(_send(c, cid, "嗨"))
    assert [e["type"] for e in ev] == ["thinking", "done"]
    msgs = _messages(c, cid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == ""


def _mid_stream(app, message):
    app.dependency_overrides[get_chat_client] = lambda: FakeChatClient(
        error=OllamaStreamError(message), error_after=2
    )


def test_mid_stream_error_emits_error_event_and_saves_nothing(make_user, app):
    _mid_stream(app, "boom")
    c = make_user("alice")
    cid = _new_conv(c)
    r = _send(c, cid, "嗨")
    assert r.status_code == 200
    ev = events(r)
    assert ev[-1] == {"type": "error", "detail": "boom"}
    assert all(e["type"] != "done" for e in ev)
    assert [m["role"] for m in _messages(c, cid)] == ["user"]


def test_mid_stream_error_with_empty_message_uses_default_detail(make_user, app):
    _mid_stream(app, "")
    c = make_user("alice")
    cid = _new_conv(c)
    assert events(_send(c, cid, "嗨"))[-1] == {"type": "error", "detail": "回覆中斷"}


def test_user_and_assistant_messages_saved(make_user, fake_chat):
    c = make_user("alice")
    cid = _new_conv(c)
    _send(c, cid, "嗨")
    msgs = _messages(c, cid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "你好"


def test_first_message_sets_title(make_user, fake_chat):
    c = make_user("alice")
    cid = _new_conv(c)
    _send(c, cid, "嗨")
    assert c.get(f"/api/conversations/{cid}").json()["title"] == "嗨"


def test_second_turn_sends_full_history(make_user, fake_chat):
    c = make_user("alice")
    cid = _new_conv(c)
    _send(c, cid, "第一句")
    _send(c, cid, "第二句")
    assert fake_chat.calls[1]["messages"] == [
        {"role": "user", "content": "第一句"},
        {"role": "assistant", "content": "你好"},
        {"role": "user", "content": "第二句"},
    ]


def test_send_to_other_users_conversation_is_404_and_no_ollama_call(make_user, fake_chat):
    alice = make_user("alice")
    bob = make_user("bob")
    cid = _new_conv(alice)
    assert _send(bob, cid, "嗨").status_code == 404
    assert fake_chat.calls == []
    assert _messages(alice, cid) == []


def test_send_to_nonexistent_conversation_404(make_user, fake_chat):
    c = make_user("alice")
    assert _send(c, 999, "嗨").status_code == 404
    assert fake_chat.calls == []


def test_blank_or_too_long_content_422(make_user, fake_chat):
    c = make_user("alice")
    cid = _new_conv(c)
    blank = _send(c, cid, "   ")
    assert blank.status_code == 422
    assert "訊息不能是空白" in blank.json()["detail"][0]["msg"]
    too_long = _send(c, cid, "字" * 8001)
    assert too_long.status_code == 422
    assert "訊息最多 8000 字" in too_long.json()["detail"][0]["msg"]
    assert _send(c, cid, "字" * 8000).status_code == 200


def test_content_stored_as_sent(make_user, fake_chat):
    c = make_user("alice")
    cid = _new_conv(c)
    _send(c, cid, "  嗨  ")
    assert _messages(c, cid)[0]["content"] == "  嗨  "


def test_ollama_unavailable_returns_503_and_keeps_user_message(make_user, app):
    app.dependency_overrides[get_chat_client] = lambda: FakeChatClient(error=OllamaUnavailable())
    c = make_user("alice")
    cid = _new_conv(c)
    r = _send(c, cid, "嗨")
    assert r.status_code == 503
    assert r.json()["detail"] == "無法連線到 Ollama"
    assert [m["role"] for m in _messages(c, cid)] == ["user"]


def test_model_not_found_returns_503_with_model_name(make_user, app):
    app.dependency_overrides[get_chat_client] = lambda: FakeChatClient(
        error=ModelNotFound("gemma4:e4b")
    )
    c = make_user("alice")
    cid = _new_conv(c)
    r = _send(c, cid, "嗨")
    assert r.status_code == 503
    assert "gemma4:e4b" in r.json()["detail"]


def test_requires_login(client):
    r = client.post("/api/conversations/1/messages", json={"content": "嗨"})
    assert r.status_code == 401


def _regen(c, cid):
    return c.post(f"/api/conversations/{cid}/regenerate")


def test_regenerate_answers_last_user_message(make_user, app):
    app.dependency_overrides[get_chat_client] = lambda: FakeChatClient(error=OllamaUnavailable())
    c = make_user("alice")
    cid = _new_conv(c)
    assert _send(c, cid, "嗨").status_code == 503
    fake = FakeChatClient()
    app.dependency_overrides[get_chat_client] = lambda: fake
    r = _regen(c, cid)
    assert r.status_code == 200
    assert events(r)[-1]["type"] == "done"
    assert [m["role"] for m in _messages(c, cid)] == ["user", "assistant"]
    assert fake.calls[-1]["messages"] == [{"role": "user", "content": "嗨"}]


def test_regenerate_409_when_already_answered(make_user, fake_chat):
    c = make_user("alice")
    cid = _new_conv(c)
    _send(c, cid, "嗨")
    r = _regen(c, cid)
    assert r.status_code == 409
    assert r.json()["detail"] == "沒有需要重新產生的回答"


def test_regenerate_409_when_no_messages(make_user, fake_chat):
    c = make_user("alice")
    cid = _new_conv(c)
    assert _regen(c, cid).status_code == 409
    assert fake_chat.calls == []


def test_regenerate_404_for_other_users_conversation(make_user, fake_chat):
    alice = make_user("alice")
    bob = make_user("bob")
    cid = _new_conv(alice)
    assert _regen(bob, cid).status_code == 404
    assert fake_chat.calls == []


def test_regenerate_503_when_unavailable(make_user, app):
    app.dependency_overrides[get_chat_client] = lambda: FakeChatClient(error=OllamaUnavailable())
    c = make_user("alice")
    cid = _new_conv(c)
    _send(c, cid, "嗨")
    r = _regen(c, cid)
    assert r.status_code == 503
    assert r.json()["detail"] == "無法連線到 Ollama"


def test_regenerate_requires_login(client):
    assert client.post("/api/conversations/1/regenerate").status_code == 401


class DeletingChatClient:
    """Deletes the conversation mid-stream, as another tab would."""

    def __init__(self, db_path):
        self.db_path = db_path

    async def stream_chat(self, messages, *, system=None, temperature=None):
        from app.db import connect

        yield ChatDelta("content", "你")
        conn = connect(self.db_path)
        conn.execute("DELETE FROM conversations")
        conn.commit()
        conn.close()
        yield ChatDelta("content", "好")


def test_save_failure_after_stream_emits_error_event(make_user, app, settings):
    app.dependency_overrides[get_chat_client] = lambda: DeletingChatClient(settings.db_path)
    c = make_user("alice")
    cid = _new_conv(c)
    ev = events(_send(c, cid, "嗨"))
    assert ev[-1] == {"type": "error", "detail": "回覆無法儲存"}
    assert all(e["type"] != "done" for e in ev)


# --- chat follows the system LLM settings ---------------------------------

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app import deps
from app.main import create_app


@pytest.fixture
def admin_chat(settings, kb_services, monkeypatch):
    """Admin client on an app using the real get_chat_client; build_chat_client is faked."""
    built: list = []
    fake = FakeChatClient()

    def build(s, llm):
        built.append(llm)
        return fake

    monkeypatch.setattr(deps, "build_chat_client", build)

    def make(**overrides):
        app = create_app(replace(settings, admin_usernames=("admin",), **overrides), kb_services)
        c = TestClient(app)
        creds = {"username": "admin", "password": "password123"}
        assert c.post("/api/auth/register", json=creds).status_code == 201
        assert c.post("/api/auth/login", json=creds).status_code == 200
        return c

    return make, built, fake


def test_chat_uses_configured_provider(admin_chat):
    make, built, _ = admin_chat
    c = make(gemini_api_key="k")
    assert c.put("/api/settings", json={"llm_provider": "gemini", "gemini_model": "g-1"}).status_code == 200
    assert _send(c, _new_conv(c), "嗨").status_code == 200
    assert built[-1].llm_provider == "gemini" and built[-1].gemini_model == "g-1"


def test_settings_change_applies_to_next_request(admin_chat):
    make, built, _ = admin_chat
    c = make()
    c.put("/api/settings", json={"ollama_model": "m1"})
    cid = _new_conv(c)
    _send(c, cid, "一")
    c.put("/api/settings", json={"ollama_model": "m2"})
    _send(c, cid, "二")
    assert [b.ollama_model for b in built] == ["m1", "m2"]


def test_system_prompt_and_temperature_are_passed(admin_chat):
    make, _, fake = admin_chat
    c = make()
    c.put("/api/settings", json={"system_prompt": "你是貓", "temperature": 0.3})
    _send(c, _new_conv(c), "嗨")
    assert fake.calls[-1]["system"] == "你是貓"
    assert fake.calls[-1]["temperature"] == 0.3
    assert fake.calls[-1]["messages"] == [{"role": "user", "content": "嗨"}]


def test_system_prompt_combines_with_knowledge_base(admin_chat):
    make, _, fake = admin_chat
    c = make()
    c.put("/api/settings", json={"system_prompt": "你是貓"})
    kb = c.post("/api/kbs", json={"name": "手冊"}).json()
    cid = _new_conv(c)
    c.patch(f"/api/conversations/{cid}", json={"kb_id": kb["id"]})
    assert _send(c, cid, "嗨").status_code == 200
    system = fake.calls[-1]["system"]
    assert system.startswith("你是貓\n\n") and "參考資料：" in system
    assert all(m["role"] != "system" for m in fake.calls[-1]["messages"])


def test_no_system_when_both_empty(admin_chat):
    make, _, fake = admin_chat
    c = make()
    _send(c, _new_conv(c), "嗨")
    assert fake.calls[-1]["system"] is None
    assert fake.calls[-1]["temperature"] is None


def test_unavailable_provider_returns_503(admin_chat, settings):
    from app.db import connect
    from app.llm_settings import save_llm_settings

    make, built, _ = admin_chat
    c = make(gemini_api_key="")
    # Saved while a key was configured; the key has since been removed.
    conn = connect(settings.db_path)
    save_llm_settings(conn, settings, {"llm_provider": "gemini"})
    conn.close()
    cid = _new_conv(c)
    r = _send(c, cid, "嗨")
    assert r.status_code == 503
    assert r.json()["detail"] == "Gemini 目前不可用：未設定 GEMINI_API_KEY"
    assert built == []
    assert _messages(c, cid) == []

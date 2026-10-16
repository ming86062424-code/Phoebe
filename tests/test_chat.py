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
    assert fake_chat.calls[1][1] == {"role": "assistant", "content": "你好"}
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
    assert fake_chat.calls[1] == [
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
    assert _send(c, cid, "   ").status_code == 422
    assert _send(c, cid, "字" * 8001).status_code == 422
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
    assert fake.calls[-1] == [{"role": "user", "content": "嗨"}]


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

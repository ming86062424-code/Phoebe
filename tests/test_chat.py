import pytest

from app.deps import get_chat_client
from app.ollama_client import ChatDelta, ModelNotFound, OllamaStreamError, OllamaUnavailable
from tests.fakes import FakeChatClient


def _new_conv(c):
    return c.post("/api/conversations").json()["id"]


def _send(c, cid, content):
    return c.post(f"/api/conversations/{cid}/messages", json={"content": content})


def _messages(c, cid):
    return c.get(f"/api/conversations/{cid}").json()["messages"]


def test_stream_returns_concatenated_text(make_user, fake_chat):
    c = make_user("alice")
    cid = _new_conv(c)
    r = _send(c, cid, "嗨")
    assert r.status_code == 200 and r.text == "你好"
    assert r.headers["content-type"].startswith("text/plain")


def test_thinking_is_not_streamed_or_saved_yet(make_user, fake_chat):
    c = make_user("alice")
    cid = _new_conv(c)
    r = _send(c, cid, "嗨")
    assert r.text == "你好"
    assert _messages(c, cid)[1]["content"] == "你好"


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


def test_mid_stream_error_does_not_save_assistant(make_user, app):
    app.dependency_overrides[get_chat_client] = lambda: FakeChatClient(
        deltas=[ChatDelta("content", "你"), ChatDelta("content", "好")], error=OllamaStreamError("boom"), error_after=1
    )
    c = make_user("alice")
    cid = _new_conv(c)
    with pytest.raises(OllamaStreamError):
        _send(c, cid, "嗨")
    assert [m["role"] for m in _messages(c, cid)] == ["user"]


def test_requires_login(client):
    r = client.post("/api/conversations/1/messages", json={"content": "嗨"})
    assert r.status_code == 401

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
        app = create_app(
            replace(settings, admin_username="admin", admin_password="password123", **overrides),
            kb_services,
        )
        c = TestClient(app)
        creds = {"username": "admin", "password": "password123"}
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


def test_store_down_during_retrieval_returns_503(admin_chat, kb_services, monkeypatch):
    from app.kb.chunk_store import KnowledgeStoreError

    def boom(*args, **kwargs):
        raise KnowledgeStoreError("無法連線到 Milvus")

    monkeypatch.setattr(kb_services.index._store, "dense_search", boom)
    make, _, _ = admin_chat
    c = make()
    kb = c.post("/api/kbs", json={"name": "手冊"}).json()
    cid = _new_conv(c)
    c.patch(f"/api/conversations/{cid}", json={"kb_id": kb["id"]})
    r = _send(c, cid, "嗨")
    assert r.status_code == 503
    assert r.json()["detail"] == "知識庫檢索失敗：無法連線到 Milvus"


# --- knowledge-base images sent to the model ------------------------------

import io
import logging

from app.kb.index import Source
from app.llm import ImagesNotSupported
from tests.pdf_factory import make_pdf

IMAGE_NOTE = "（圖片未提供給模型）"


def _png_bytes(size, color):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _ref(name, size=(4, 3)):
    return {"name": name, "width": size[0], "height": size[1]}


def _kb_chat_with_images(c, kb_services, monkeypatch, layout):
    """A conversation on a knowledge base whose search returns `layout`: a list of
    (document index, page_image name or None, [image names]) in hybrid score order.

    Returns (cid, kb_id, document ids, {(document index, name): bytes written}).
    """
    kb_id = c.post("/api/kbs", json={"name": "手冊"}).json()["id"]
    docs = []
    for i in range(1 + max(d for d, _, _ in layout)):
        r = c.post(
            f"/api/kbs/{kb_id}/documents",
            files={"file": (f"d{i}.pdf", make_pdf(["x"]), "application/pdf")},
            data={"parser": "fake"},
        )
        docs.append(r.json()["id"])
    colors = iter(["red", "green", "blue", "yellow", "white", "black", "gray", "orange"])
    files = {}
    sources = []
    for n, (d, page_image, images) in enumerate(layout, start=1):
        for name in ([page_image] if page_image else []) + images:
            if (d, name) in files:  # the same image cited by another source
                continue
            data = _png_bytes((4 + n, 3), next(colors))
            files[(d, name)] = data
            (kb_services.doc_dir(kb_id, docs[d]) / name).write_bytes(data)
        sources.append(
            Source(
                document_id=docs[d],
                filename=f"d{d}.pdf",
                page=n,
                score=1 / n,
                text=f"內容 {n}",
                kind="page_image" if page_image else "text",
                images=[_ref(x) for x in images],
                page_image=_ref(page_image) if page_image else None,
            )
        )
    monkeypatch.setattr(kb_services.index, "search", lambda *a, **k: list(sources))
    cid = _new_conv(c)
    c.patch(f"/api/conversations/{cid}", json={"kb_id": kb_id})
    return cid, kb_id, docs, files


LAYOUT = [
    (0, "page_1.png", ["img_1_1.png", "img_1_2.png"]),
    (1, None, []),
    (1, "page_3.png", ["img_3_1.png"]),
]


def test_kb_answer_sends_source_images_in_order_up_to_the_limit(
    make_user, fake_chat, kb_services, settings, monkeypatch
):
    from app.db import connect

    c = make_user("alice")
    cid, _, _, files = _kb_chat_with_images(c, kb_services, monkeypatch, LAYOUT)
    ev = events(_send(c, cid, "圖片 裡 有 什麼"))
    assert ev[0]["type"] == "sources" and ev[-1]["type"] == "done"
    sent = fake_chat.calls[0]["messages"]
    assert sent[-1]["images"] == [
        files[(0, "page_1.png")],
        files[(0, "img_1_1.png")],
        files[(0, "img_1_2.png")],
        files[(1, "page_3.png")],
    ]
    system = fake_chat.calls[0]["system"]
    assert "附圖依序對應參考資料 [編號]" in system
    for line in ("附圖 1 → [1]", "附圖 2 → [1]", "附圖 3 → [1]", "附圖 4 → [3]"):
        assert line in system
    assert "附圖 5" not in system and IMAGE_NOTE not in system
    # the sources event is unchanged: plain dicts with the new fields
    assert ev[0]["sources"][0]["page_image"] == _ref("page_1.png")
    # nothing about the images is kept: the next turn's history has no image bytes
    _send(c, cid, "再問")
    assert all("images" not in m for m in fake_chat.calls[1]["messages"][:-1])
    conn = connect(settings.db_path)
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(messages)")]
        rows = conn.execute("SELECT content, sources FROM messages").fetchall()
    finally:
        conn.close()
    assert "images" not in cols
    assert all("iVBOR" not in (r[0] + (r[1] or "")) for r in rows)  # no base64 PNG


def test_sources_carry_their_knowledge_base_id(make_user, fake_chat, kb_services, monkeypatch):
    """The UI builds image URLs from it, even after the conversation switches knowledge base."""
    c = make_user("alice")
    cid, kb_id, _, _ = _kb_chat_with_images(c, kb_services, monkeypatch, LAYOUT)
    ev = events(_send(c, cid, "圖片"))
    assert [s["kb_id"] for s in ev[0]["sources"]] == [kb_id] * len(LAYOUT)
    c.patch(f"/api/conversations/{cid}", json={"kb_id": None})
    saved = [m for m in _messages(c, cid) if m["role"] == "assistant"][0]
    assert [s["kb_id"] for s in saved["sources"]] == [kb_id] * len(LAYOUT)


def test_kb_max_images_limits_and_zero_disables(settings, kb_services, monkeypatch):
    for limit in (2, 0):
        app = create_app(replace(settings, kb_max_images=limit), kb_services)
        fake = FakeChatClient()
        app.dependency_overrides[get_chat_client] = lambda: fake
        c = TestClient(app)
        creds = {"username": f"user{limit}", "password": "password123"}
        assert c.post("/api/auth/register", json=creds).status_code == 201
        assert c.post("/api/auth/login", json=creds).status_code == 200
        cid, *_ = _kb_chat_with_images(c, kb_services, monkeypatch, LAYOUT)
        _send(c, cid, "嗨")
        last = fake.calls[0]["messages"][-1]
        assert len(last.get("images", [])) == limit
        assert ("附圖依序對應參考資料" in fake.calls[0]["system"]) == bool(limit)
        if limit == 0:
            assert last == {"role": "user", "content": "嗨"}


def test_missing_image_files_are_skipped_with_a_log(
    make_user, fake_chat, kb_services, monkeypatch, caplog
):
    c = make_user("alice")
    cid, kb_id, docs, files = _kb_chat_with_images(c, kb_services, monkeypatch, LAYOUT)
    (kb_services.doc_dir(kb_id, docs[0]) / "img_1_1.png").unlink()
    with caplog.at_level(logging.INFO):
        ev = events(_send(c, cid, "嗨"))
    assert ev[-1]["type"] == "done"
    assert fake_chat.calls[0]["messages"][-1]["images"] == [
        files[(0, "page_1.png")],
        files[(0, "img_1_2.png")],
        files[(1, "page_3.png")],
        files[(1, "img_3_1.png")],
    ]
    system = fake_chat.calls[0]["system"]
    assert "附圖 2 → [1]" in system and "附圖 3 → [3]" in system and "附圖 4 → [3]" in system
    assert "img_1_1.png" in caplog.text


def test_large_images_are_downscaled_before_sending(make_user, fake_chat, kb_services, monkeypatch):
    from PIL import Image

    c = make_user("alice")
    cid, kb_id, docs, files = _kb_chat_with_images(
        c, kb_services, monkeypatch, [(0, None, ["img_1_1.png", "img_1_2.png"])]
    )
    big = io.BytesIO()
    Image.new("RGB", (3072, 1000), "red").save(big, format="PNG")
    (kb_services.doc_dir(kb_id, docs[0]) / "img_1_1.png").write_bytes(big.getvalue())
    _send(c, cid, "嗨")
    sent = fake_chat.calls[0]["messages"][-1]["images"]
    with Image.open(io.BytesIO(sent[0])) as im:
        assert (im.format, im.size) == ("PNG", (1536, 500))
    assert sent[1] == files[(0, "img_1_2.png")]  # small images are passed through untouched


def test_kb_answer_without_images_sends_no_images_key(make_user, fake_chat, kb_services, monkeypatch):
    c = make_user("alice")
    cid, *_ = _kb_chat_with_images(c, kb_services, monkeypatch, [(0, None, [])])
    _send(c, cid, "嗨")
    assert fake_chat.calls[0]["messages"][-1] == {"role": "user", "content": "嗨"}
    assert "附圖" not in fake_chat.calls[0]["system"]


# --- retry without images ---------------------------------------------------


class _RejectingGeminiModels:
    """Gemini SDK stand-in that rejects any request carrying an image part."""

    def __init__(self):
        self.calls = []

    async def generate_content_stream(self, *, model, contents, config=None):
        from google.genai import errors, types

        self.calls.append({"contents": contents, "config": config})
        has_image = any(p.inline_data is not None for c in contents for p in c.parts)

        async def gen():
            if has_image:
                raise errors.ClientError(
                    400,
                    {
                        "error": {
                            "code": 400,
                            "message": "Image input modality is not enabled for models/x",
                            "status": "INVALID_ARGUMENT",
                        }
                    },
                )
            yield types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(role="model", parts=[types.Part(text="純文字回答")])
                    )
                ]
            )

        return gen()


def _rejecting_ollama(seen):
    import httpx

    from app.llm import OllamaChatClient

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if any("images" in m for m in body["messages"]):
            return httpx.Response(
                500, json={"error": "this model is missing data required for image input"}
            )
        lines = [{"message": {"content": "純文字回答"}, "done": False}, {"done": True}]
        return httpx.Response(200, content="".join(json.dumps(x) + "\n" for x in lines).encode())

    return OllamaChatClient("http://ollama.test", "text-only", transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("provider", ["ollama", "gemini"])
def test_retries_without_images_when_model_rejects_them(
    provider, make_user, app, kb_services, monkeypatch
):
    from types import SimpleNamespace

    from app.llm.gemini import GeminiChatClient

    if provider == "ollama":
        seen: list = []
        client = _rejecting_ollama(seen)

        def requests():
            return [
                (any("images" in m for m in b["messages"]), b["messages"][0]["content"])
                for b in seen
            ]
    else:
        models = _RejectingGeminiModels()
        client = GeminiChatClient("k", "x", client=SimpleNamespace(aio=SimpleNamespace(models=models)))

        def requests():
            return [
                (
                    any(p.inline_data is not None for c in call["contents"] for p in c.parts),
                    call["config"].system_instruction,
                )
                for call in models.calls
            ]

    app.dependency_overrides[get_chat_client] = lambda: client
    c = make_user("alice")
    cid, *_ = _kb_chat_with_images(c, kb_services, monkeypatch, LAYOUT)
    r = _send(c, cid, "圖裡有什麼")
    assert r.status_code == 200
    ev = events(r)
    assert [e["type"] for e in ev] == ["sources", "content", "done"]
    assert ev[1]["text"] == "純文字回答"
    sent = requests()
    assert [with_images for with_images, _ in sent] == [True, False]
    assert IMAGE_NOTE not in sent[0][1]
    assert sent[1][1].endswith(IMAGE_NOTE)
    assert _messages(c, cid)[1]["content"] == "純文字回答"


def test_retry_without_images_happens_only_once(make_user, app, kb_services, monkeypatch):
    fake = FakeChatClient(error=ImagesNotSupported("model does not support images"))
    app.dependency_overrides[get_chat_client] = lambda: fake
    c = make_user("alice")
    cid, *_ = _kb_chat_with_images(c, kb_services, monkeypatch, LAYOUT)
    r = _send(c, cid, "嗨")
    assert r.status_code == 503
    assert len(fake.calls) == 2
    assert len(fake.calls[0]["messages"][-1]["images"]) == 4
    assert "images" not in fake.calls[1]["messages"][-1]
    assert fake.calls[1]["system"].endswith(IMAGE_NOTE)


def test_no_retry_once_content_has_been_streamed(make_user, app, kb_services, monkeypatch):
    fake = FakeChatClient(
        deltas=[ChatDelta("content", "部分")],
        error=ImagesNotSupported("model does not support images"),
        error_after=1,
    )
    app.dependency_overrides[get_chat_client] = lambda: fake
    c = make_user("alice")
    cid, *_ = _kb_chat_with_images(c, kb_services, monkeypatch, LAYOUT)
    ev = events(_send(c, cid, "嗨"))
    assert [e["type"] for e in ev] == ["sources", "content", "error"]
    assert len(fake.calls) == 1


def test_no_retry_when_no_images_were_sent(make_user, app):
    fake = FakeChatClient(error=ImagesNotSupported("model does not support images"))
    app.dependency_overrides[get_chat_client] = lambda: fake
    c = make_user("alice")
    assert _send(c, _new_conv(c), "嗨").status_code == 503
    assert len(fake.calls) == 1


def test_an_image_cited_by_several_sources_is_sent_once(make_user, fake_chat, kb_services, monkeypatch):
    c = make_user("alice")
    layout = [
        (0, "page_1.png", ["img_1_1.png"]),
        (0, "page_1.png", ["img_1_1.png", "img_1_2.png"]),
        (1, "page_1.png", []),  # same name, another document: a different image
    ]
    cid, _, _, files = _kb_chat_with_images(c, kb_services, monkeypatch, layout)
    _send(c, cid, "嗨")
    assert fake_chat.calls[0]["messages"][-1]["images"] == [
        files[(0, "page_1.png")],
        files[(0, "img_1_1.png")],
        files[(0, "img_1_2.png")],
        files[(1, "page_1.png")],
    ]
    system = fake_chat.calls[0]["system"]
    for line in ("附圖 1 → [1]", "附圖 2 → [1]", "附圖 3 → [2]", "附圖 4 → [3]"):
        assert line in system
    assert "附圖 5" not in system

import json

from app.kb.parsers.base import PageText
from tests.pdf_factory import make_pdf


def _create_kb(client, name="手冊"):
    r = client.post("/api/kbs", json={"name": name})
    assert r.status_code == 201
    return r.json()


def _upload(client, kb_id, parser="fake", content=None, filename="a.pdf"):
    content = make_pdf(["hello"]) if content is None else content
    return client.post(
        f"/api/kbs/{kb_id}/documents",
        files={"file": (filename, content, "application/pdf")},
        data={"parser": parser},
    )


def test_requires_login(client):
    assert client.get("/api/kbs").status_code == 401
    assert client.get("/api/parsers").status_code == 401


def test_parsers_listed(make_user):
    alice = make_user("alice")
    ids = [p["id"] for p in alice.get("/api/parsers").json()]
    assert ids == ["liteparse", "fake"]


def test_kb_crud_and_isolation(make_user):
    alice, bob = make_user("alice"), make_user("bob")
    kb = _create_kb(alice)
    assert alice.get("/api/kbs").json()[0]["document_count"] == 0
    assert bob.get("/api/kbs").json() == []
    assert bob.get(f"/api/kbs/{kb['id']}/documents").status_code == 404
    assert bob.delete(f"/api/kbs/{kb['id']}").status_code == 404
    r = alice.patch(f"/api/kbs/{kb['id']}", json={"name": " 新名稱 "})
    assert r.json()["name"] == "新名稱"
    assert alice.post("/api/kbs", json={"name": "  "}).status_code == 422
    assert alice.delete(f"/api/kbs/{kb['id']}").status_code == 204
    assert alice.get("/api/kbs").json() == []


def test_upload_parses_and_indexes(make_user, fake_parser, kb_services):
    alice = make_user("alice")
    kb = _create_kb(alice)
    r = _upload(alice, kb["id"])
    assert r.status_code == 201
    doc = r.json()
    assert (doc["status"], doc["chunk_count"], doc["filename"]) == ("ready", 2, "a.pdf")
    assert kb_services.file_path(kb["id"], doc["id"]).is_file()
    hits = alice.get(f"/api/kbs/{kb['id']}/search", params={"q": "cherry"}).json()
    assert hits[0]["page"] == 2 and hits[0]["filename"] == "a.pdf"


def test_upload_with_real_liteparse(make_user):
    alice = make_user("alice")
    kb = _create_kb(alice)
    pdf = make_pdf(["Quarterly revenue grew", "Office moved to Taipei"])
    doc = _upload(alice, kb["id"], parser="liteparse", content=pdf).json()
    assert doc["status"] == "ready"
    hits = alice.get(f"/api/kbs/{kb['id']}/search", params={"q": "Taipei office"}).json()
    assert hits[0]["page"] == 2


def test_upload_validation(make_user, settings):
    alice = make_user("alice")
    kb = _create_kb(alice)
    assert _upload(alice, kb["id"], filename="a.txt").status_code == 422
    assert _upload(alice, kb["id"], content=b"not a pdf").status_code == 422
    assert _upload(alice, kb["id"], parser="nope").status_code == 422
    assert alice.get(f"/api/kbs/{kb['id']}/documents").json() == []


def test_upload_rejects_unavailable_parser(make_user, fake_parser):
    alice = make_user("alice")
    kb = _create_kb(alice)
    fake_parser.available = False
    r = _upload(alice, kb["id"])
    assert r.status_code == 422 and r.json()["detail"] == "未安裝"


def test_failed_document_can_be_retried(make_user, fake_parser):
    from app.kb.parsers.base import ParseError

    alice = make_user("alice")
    kb = _create_kb(alice)
    fake_parser.error = ParseError("爆了")
    doc = _upload(alice, kb["id"]).json()
    assert (doc["status"], doc["error"]) == ("failed", "爆了")
    fake_parser.error = None
    r = alice.post(f"/api/kbs/{kb['id']}/documents/{doc['id']}/retry")
    assert r.json()["status"] == "ready"
    assert alice.post(f"/api/kbs/{kb['id']}/documents/{doc['id']}/retry").status_code == 409


def test_delete_document_removes_file_and_vectors(make_user, kb_services):
    alice = make_user("alice")
    kb = _create_kb(alice)
    doc = _upload(alice, kb["id"]).json()
    assert alice.delete(f"/api/kbs/{kb['id']}/documents/{doc['id']}").status_code == 204
    assert not kb_services.file_path(kb["id"], doc["id"]).exists()
    assert alice.get(f"/api/kbs/{kb['id']}/search", params={"q": "apple"}).json() == []
    assert alice.delete(f"/api/kbs/{kb['id']}/documents/{doc['id']}").status_code == 404


def _stream_events(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def test_conversation_kb_selection(make_user):
    alice, bob = make_user("alice"), make_user("bob")
    kb = _create_kb(alice)
    bob_kb = _create_kb(bob)
    cid = alice.post("/api/conversations").json()["id"]
    r = alice.patch(f"/api/conversations/{cid}", json={"kb_id": kb["id"]})
    assert r.status_code == 200 and r.json()["kb_id"] == kb["id"]
    assert alice.patch(f"/api/conversations/{cid}", json={"kb_id": bob_kb["id"]}).status_code == 404
    assert alice.patch(f"/api/conversations/{cid}", json={"kb_id": None}).json()["kb_id"] is None
    # Deleting the knowledge base detaches it from conversations.
    alice.patch(f"/api/conversations/{cid}", json={"kb_id": kb["id"]})
    alice.delete(f"/api/kbs/{kb['id']}")
    assert alice.get(f"/api/conversations/{cid}").json()["kb_id"] is None


def test_chat_with_kb_streams_and_saves_sources(make_user, fake_chat, fake_parser):
    fake_parser.pages = [PageText(1, "特休 十四 天"), PageText(3, "rocket fuel")]
    alice = make_user("alice")
    kb = _create_kb(alice, "員工手冊")
    _upload(alice, kb["id"])
    cid = alice.post("/api/conversations").json()["id"]
    alice.patch(f"/api/conversations/{cid}", json={"kb_id": kb["id"]})

    r = alice.post(f"/api/conversations/{cid}/messages", json={"content": "rocket fuel"})
    events = _stream_events(r)
    assert events[0]["type"] == "sources"
    assert events[0]["sources"][0]["page"] == 3
    system = fake_chat.calls[0][0]
    assert system["role"] == "system"
    assert "員工手冊" in system["content"] and "rocket fuel" in system["content"]
    assert fake_chat.calls[0][-1] == {"role": "user", "content": "rocket fuel"}

    saved = alice.get(f"/api/conversations/{cid}").json()["messages"]
    assert saved[0]["sources"] is None
    assert saved[1]["sources"][0]["page"] == 3


def test_chat_without_kb_has_no_sources_event(make_user, fake_chat):
    alice = make_user("alice")
    cid = alice.post("/api/conversations").json()["id"]
    events = _stream_events(alice.post(f"/api/conversations/{cid}/messages", json={"content": "hi"}))
    assert all(e["type"] != "sources" for e in events)
    assert fake_chat.calls[0][0]["role"] == "user"


def test_regenerate_with_kb_retrieves_again(make_user, fake_chat):
    from app.ollama_client import OllamaUnavailable

    alice = make_user("alice")
    kb = _create_kb(alice)
    _upload(alice, kb["id"])
    cid = alice.post("/api/conversations").json()["id"]
    alice.patch(f"/api/conversations/{cid}", json={"kb_id": kb["id"]})
    fake_chat.error = OllamaUnavailable()
    assert alice.post(f"/api/conversations/{cid}/messages", json={"content": "apple"}).status_code == 503
    fake_chat.error = None
    events = _stream_events(alice.post(f"/api/conversations/{cid}/regenerate"))
    assert events[0]["type"] == "sources" and events[-1]["type"] == "done"


def test_chat_retrieval_failure_returns_503(make_user, fake_chat, kb_services, monkeypatch):
    alice = make_user("alice")
    kb = _create_kb(alice)
    cid = alice.post("/api/conversations").json()["id"]
    alice.patch(f"/api/conversations/{cid}", json={"kb_id": kb["id"]})

    def boom(*args):
        raise ConnectionError("ollama down")

    monkeypatch.setattr(kb_services.index, "search", boom)
    r = alice.post(f"/api/conversations/{cid}/messages", json={"content": "x"})
    assert r.status_code == 503 and "知識庫檢索失敗" in r.json()["detail"]


def test_system_prompt_gets_full_chunk_text_not_the_snippet(make_user, fake_chat, fake_parser):
    filler = "intro " * 90  # one chunk, but the answer sits past the 400-character snippet
    fake_parser.pages = [PageText(1, filler + "密碼 每 九十 天 更換")]
    alice = make_user("alice")
    kb = _create_kb(alice)
    _upload(alice, kb["id"])
    cid = alice.post("/api/conversations").json()["id"]
    alice.patch(f"/api/conversations/{cid}", json={"kb_id": kb["id"]})
    events = _stream_events(
        alice.post(f"/api/conversations/{cid}/messages", json={"content": "密碼 更換"})
    )
    assert "九十" in fake_chat.calls[0][0]["content"]
    # the UI and the database still get the short snippet
    shown = events[0]["sources"][0]["text"]
    assert len(shown) <= 400 and "九十" not in shown


def _locked(monkeypatch, target):
    """Make deleting `target` fail like Windows does while OCR holds the file open."""
    import pathlib
    import shutil

    real_unlink, real_rmtree = pathlib.Path.unlink, shutil.rmtree

    def unlink(self, missing_ok=False):
        if self == target:
            raise PermissionError(13, "file in use", str(self))
        return real_unlink(self, missing_ok=missing_ok)

    def rmtree(path, ignore_errors=False, **kwargs):
        if target.is_relative_to(path):
            if ignore_errors:
                return  # like the real one: silently leaves the files behind
            raise PermissionError(13, "file in use", str(path))
        return real_rmtree(path, ignore_errors=ignore_errors, **kwargs)

    monkeypatch.setattr(pathlib.Path, "unlink", unlink)
    monkeypatch.setattr(shutil, "rmtree", rmtree)


def test_delete_document_succeeds_when_file_is_locked(make_user, kb_services, monkeypatch):
    alice = make_user("alice")
    kb = _create_kb(alice)
    doc = _upload(alice, kb["id"]).json()
    path = kb_services.file_path(kb["id"], doc["id"])
    _locked(monkeypatch, path)
    assert alice.delete(f"/api/kbs/{kb['id']}/documents/{doc['id']}").status_code == 204
    assert alice.get(f"/api/kbs/{kb['id']}/search", params={"q": "apple"}).json() == []
    assert path.exists()
    monkeypatch.undo()
    kb_services.worker.cleanup()
    assert not path.exists()


def test_delete_kb_succeeds_when_a_file_is_locked(make_user, kb_services, monkeypatch):
    alice = make_user("alice")
    kb = _create_kb(alice)
    doc = _upload(alice, kb["id"]).json()
    folder = kb_services.kb_dir / str(kb["id"])
    _locked(monkeypatch, kb_services.file_path(kb["id"], doc["id"]))
    monkeypatch.setattr(kb_services.index, "delete_kb", lambda kb_id: deleted.append(kb_id))
    deleted: list[int] = []
    assert alice.delete(f"/api/kbs/{kb['id']}").status_code == 204
    assert deleted == [kb["id"]]
    monkeypatch.undo()
    kb_services.worker.cleanup()
    assert not folder.exists()

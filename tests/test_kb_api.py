import json

from app.kb.parsers.base import ParsedChunk
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
    assert _upload(alice, kb["id"], filename="a.docx").status_code == 422
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
    fake_parser.pages = [ParsedChunk(1, "特休 十四 天"), ParsedChunk(3, "rocket fuel")]
    alice = make_user("alice")
    kb = _create_kb(alice, "員工手冊")
    _upload(alice, kb["id"])
    cid = alice.post("/api/conversations").json()["id"]
    alice.patch(f"/api/conversations/{cid}", json={"kb_id": kb["id"]})

    r = alice.post(f"/api/conversations/{cid}/messages", json={"content": "rocket fuel"})
    events = _stream_events(r)
    assert events[0]["type"] == "sources"
    assert events[0]["sources"][0]["page"] == 3
    system = fake_chat.calls[0]["system"]
    assert "員工手冊" in system and "rocket fuel" in system
    assert fake_chat.calls[0]["messages"][-1] == {"role": "user", "content": "rocket fuel"}

    saved = alice.get(f"/api/conversations/{cid}").json()["messages"]
    assert saved[0]["sources"] is None
    assert saved[1]["sources"][0]["page"] == 3


def test_chat_without_kb_has_no_sources_event(make_user, fake_chat):
    alice = make_user("alice")
    cid = alice.post("/api/conversations").json()["id"]
    events = _stream_events(alice.post(f"/api/conversations/{cid}/messages", json={"content": "hi"}))
    assert all(e["type"] != "sources" for e in events)
    assert fake_chat.calls[0]["messages"][0]["role"] == "user"


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
    fake_parser.pages = [ParsedChunk(1, filler + "密碼 每 九十 天 更換")]
    alice = make_user("alice")
    kb = _create_kb(alice)
    _upload(alice, kb["id"])
    cid = alice.post("/api/conversations").json()["id"]
    alice.patch(f"/api/conversations/{cid}", json={"kb_id": kb["id"]})
    events = _stream_events(
        alice.post(f"/api/conversations/{cid}/messages", json={"content": "密碼 更換"})
    )
    assert "九十" in fake_chat.calls[0]["system"]
    # the UI and the database get the whole chunk too; the client collapses it
    shown = events[0]["sources"][0]["text"]
    assert shown == filler + "密碼 每 九十 天 更換"


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


def test_oversized_upload_is_rejected_before_the_body_is_parsed(make_user, app, settings, monkeypatch):
    import dataclasses

    from starlette.requests import Request

    alice = make_user("alice")
    kb = _create_kb(alice)
    app.state.settings = dataclasses.replace(settings, max_upload_mb=0)

    async def no_parsing(self, *args, **kwargs):
        raise AssertionError("the body should not be parsed")

    monkeypatch.setattr(Request, "form", no_parsing)
    r = _upload(alice, kb["id"], content=b"%PDF-" + b"x" * 300_000)
    assert r.status_code == 413
    assert alice.get(f"/api/kbs/{kb['id']}/documents").json() == []


def test_upload_does_blocking_work_off_the_event_loop(make_user, monkeypatch):
    import asyncio

    from app.kb import store as kb_store

    calls = []
    real = kb_store.create_document

    def create_document(*args, **kwargs):
        try:
            asyncio.get_running_loop()
            calls.append("event loop")
        except RuntimeError:
            calls.append("worker thread")
        return real(*args, **kwargs)

    monkeypatch.setattr(kb_store, "create_document", create_document)
    alice = make_user("alice")
    kb = _create_kb(alice)
    assert _upload(alice, kb["id"]).status_code == 201
    assert calls == ["worker thread"]


def test_external_worker_mode_does_not_start_worker(settings, kb_services):
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from app.main import create_app

    started = []
    kb_services.worker.start = lambda: started.append("start")
    kb_services.worker.stop = lambda: started.append("stop")
    with TestClient(create_app(replace(settings, kb_worker_mode="external"), kb_services)):
        pass
    assert started == []
    with TestClient(create_app(settings, kb_services)):
        pass
    assert started == ["start", "stop"]


def test_upload_creates_pending_row_and_queues_after_file_is_in_place(make_user, kb_services, monkeypatch):
    from app.kb import store as kb_store

    alice = make_user("alice")
    kb = _create_kb(alice)
    seen = []
    real = kb_store.update_document

    def spy(conn, doc_id, **fields):
        if fields.get("status") == "queued":
            seen.append(
                (kb_store.get_document(conn, doc_id)["status"], kb_services.file_path(kb["id"], doc_id).exists())
            )
        return real(conn, doc_id, **fields)

    monkeypatch.setattr(kb_store, "update_document", spy)
    assert _upload(alice, kb["id"]).status_code == 201
    assert seen == [("pending", True)]


def test_external_mode_upload_uses_published_status(tmp_path, fake_parser):
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.db import connect, init_db, now_iso
    from app.kb.services import build_kb_services
    from app.main import create_app
    from tests.fakes_embedder import FakeEmbedder
    from tests.fakes_store import InMemoryChunkStore

    settings = Settings(
        db_path=str(tmp_path / "e.db"),
        frontend_dist=str(tmp_path / "no-dist"),
        kb_dir=str(tmp_path / "kb"),
        kb_worker_mode="external",
    )
    fake_parser.available = False  # the web process itself could not use it
    services = build_kb_services(
        settings, embedder=FakeEmbedder(),
        parsers=[fake_parser],
        threaded=False,
        store=InMemoryChunkStore(),
    )
    c = TestClient(create_app(settings, services))
    creds = {"username": "eve", "password": "password123"}
    c.post("/api/auth/register", json=creds)
    c.post("/api/auth/login", json=creds)
    kb = c.post("/api/kbs", json={"name": "k"}).json()
    r = _upload(c, kb["id"])
    assert r.status_code == 422
    assert "worker" in r.json()["detail"]
    conn = connect(settings.db_path)
    init_db(conn)
    conn.execute(
        "INSERT INTO parser_status (parser_id, available, reason, updated_at) VALUES ('fake', 1, '', ?)",
        (now_iso(),),
    )
    conn.commit()
    conn.close()
    assert c.get("/api/parsers").json()[0]["available"] is True
    assert _upload(c, kb["id"]).status_code == 201


def _store_down(monkeypatch, kb_services, *names):
    from app.kb.chunk_store import KnowledgeStoreError

    def boom(*args, **kwargs):
        raise KnowledgeStoreError("無法連線到 Milvus")

    for name in names:
        monkeypatch.setattr(kb_services.index._store, name, boom)


def test_delete_document_succeeds_when_store_is_down(make_user, kb_services, monkeypatch):
    alice = make_user("alice")
    kb = _create_kb(alice)
    doc = _upload(alice, kb["id"]).json()
    removed = []
    monkeypatch.setattr(kb_services.worker, "remove_path", lambda p: removed.append(p))
    _store_down(monkeypatch, kb_services, "delete_document")
    assert alice.delete(f"/api/kbs/{kb['id']}/documents/{doc['id']}").status_code == 204
    assert removed == [
        kb_services.file_path(kb["id"], doc["id"]),
        kb_services.doc_dir(kb["id"], doc["id"]),
    ]
    assert alice.delete(f"/api/kbs/{kb['id']}/documents/{doc['id']}").status_code == 404


def test_delete_kb_succeeds_when_store_is_down(make_user, kb_services, monkeypatch):
    alice = make_user("alice")
    kb = _create_kb(alice)
    removed = []
    monkeypatch.setattr(kb_services.worker, "remove_path", lambda p: removed.append(p))
    _store_down(monkeypatch, kb_services, "delete_kb")
    assert alice.delete(f"/api/kbs/{kb['id']}").status_code == 204
    assert removed == [kb_services.kb_dir / str(kb["id"])]
    assert kb["id"] not in [k["id"] for k in alice.get("/api/kbs").json()]


def test_deleted_document_is_never_retrieved_after_a_store_outage(make_user, kb_services, fake_chat, monkeypatch):
    alice = make_user("alice")
    kb = _create_kb(alice)
    gone = _upload(alice, kb["id"]).json()
    _store_down(monkeypatch, kb_services, "delete_document")
    assert alice.delete(f"/api/kbs/{kb['id']}/documents/{gone['id']}").status_code == 204
    # the chunks are still in the store, but the document no longer exists
    assert kb_services.index._store.rows
    assert alice.get(f"/api/kbs/{kb['id']}/search", params={"q": "apple"}).json() == []
    cid = alice.post("/api/conversations").json()["id"]
    alice.patch(f"/api/conversations/{cid}", json={"kb_id": kb["id"]})
    events = _stream_events(alice.post(f"/api/conversations/{cid}/messages", json={"content": "apple"}))
    sources = next(e for e in events if e["type"] == "sources")
    assert sources["sources"] == [] if "sources" in sources else True
    assert gone["filename"] not in fake_chat.calls[-1]["system"]


# ---- file types ----

UNSUPPORTED = "不支援的檔案類型（支援 PDF、txt、md、png、jpg、jpeg、webp）"
UNREADABLE = "無法讀取檔案內容"


def _png(fmt="PNG", size=(4, 3)):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format=fmt)
    return buf.getvalue()


def test_unsupported_extension_is_rejected(make_user):
    alice = make_user("alice")
    kb = _create_kb(alice)
    for name in ("a.docx", "a", "a.pdf.exe", "a.gif"):
        r = _upload(alice, kb["id"], filename=name)
        assert (r.status_code, r.json()["detail"]) == (422, UNSUPPORTED), name
    assert alice.get(f"/api/kbs/{kb['id']}/documents").json() == []


def test_pdf_without_magic_is_unreadable(make_user):
    alice = make_user("alice")
    kb = _create_kb(alice)
    r = _upload(alice, kb["id"], content=b"not a pdf")
    assert (r.status_code, r.json()["detail"]) == (422, UNREADABLE)


def test_text_upload_encodings(make_user, kb_services):
    alice = make_user("alice")
    kb = _create_kb(alice)
    utf8 = _upload(alice, kb["id"], filename="a.txt", content="柴犬 是 日本 的 狗".encode("utf-8"))
    assert utf8.status_code == 201
    assert (utf8.json()["status"], utf8.json()["parser"]) == ("ready", "text")
    bom = _upload(alice, kb["id"], filename="b.TXT", content="\ufeff有 BOM 的 檔案".encode("utf-8"))
    assert bom.status_code == 201 and bom.json()["status"] == "ready"
    big5 = _upload(alice, kb["id"], filename="c.txt", content="中文內容".encode("big5"))
    assert (big5.status_code, big5.json()["detail"]) == (422, UNREADABLE)
    names = [d["filename"] for d in alice.get(f"/api/kbs/{kb['id']}/documents").json()]
    assert sorted(names) == ["a.txt", "b.TXT"]
    hits = alice.get(f"/api/kbs/{kb['id']}/search", params={"q": "柴犬"}).json()
    assert hits[0]["page"] == 0 and hits[0]["kind"] == "text" and hits[0]["filename"] == "a.txt"


def test_text_files_use_the_text_parser_whatever_was_requested(make_user, kb_services, fake_parser):
    alice = make_user("alice")
    kb = _create_kb(alice)
    md = "# 一\n\n甲 乙\n\n# 二\n\n丙 丁".encode("utf-8")
    fake_parser.available = False  # the requested parser is irrelevant for text files
    doc = _upload(alice, kb["id"], parser="fake", filename="notes.md", content=md).json()
    assert (doc["parser"], doc["status"], doc["chunk_count"]) == ("text", "ready", 2)
    assert fake_parser.parsed == []
    stored = kb_services.file_path(kb["id"], doc["id"], "notes.md")
    assert stored.name == f"{doc['id']}.md" and stored.read_bytes() == md


def test_text_parser_cannot_be_requested_for_a_pdf(make_user):
    alice = make_user("alice")
    kb = _create_kb(alice)
    r = _upload(alice, kb["id"], parser="text")
    assert (r.status_code, r.json()["detail"]) == (422, "未知的解析器")
    ids = [p["id"] for p in alice.get("/api/parsers").json()]
    assert "text" not in ids


def test_image_upload_is_checked_and_keeps_its_extension(make_user, kb_services, fake_parser):
    alice = make_user("alice")
    kb = _create_kb(alice)
    for name, fmt in (("a.png", "PNG"), ("b.JPG", "JPEG"), ("c.jpeg", "JPEG"), ("d.webp", "WEBP")):
        r = _upload(alice, kb["id"], filename=name, content=_png(fmt))
        assert r.status_code == 201, name
        doc = r.json()
        assert doc["parser"] == "fake"
        path = kb_services.file_path(kb["id"], doc["id"], name)
        assert path.suffix == "." + name.rsplit(".", 1)[1].lower() and path.is_file()
        assert fake_parser.parsed[-1] == str(path)
    bad = _upload(alice, kb["id"], filename="e.png", content=b"\x89PNG\r\n\x1a\nbroken")
    assert (bad.status_code, bad.json()["detail"]) == (422, UNREADABLE)
    assert _upload(alice, kb["id"], filename="f.png", content=make_pdf(["x"])).status_code == 422


def test_long_filename_keeps_its_extension(make_user, kb_services):
    alice = make_user("alice")
    kb = _create_kb(alice)
    name = "長" * 300 + ".txt"
    doc = _upload(alice, kb["id"], filename=name, content=b"hello").json()
    assert doc["filename"].endswith(".txt") and len(doc["filename"]) <= 200
    assert doc["status"] == "ready"


def test_parser_gets_a_per_document_folder_removed_with_the_document(make_user, kb_services, fake_parser):
    alice = make_user("alice")
    kb = _create_kb(alice)
    doc = _upload(alice, kb["id"]).json()
    folder = kb_services.doc_dir(kb["id"], doc["id"])
    assert folder == kb_services.kb_dir / str(kb["id"]) / str(doc["id"])
    assert fake_parser.out_dirs == [folder] and folder.is_dir()
    (folder / "page_1.png").write_bytes(b"x")
    assert alice.delete(f"/api/kbs/{kb['id']}/documents/{doc['id']}").status_code == 204
    assert not folder.exists()
    assert not kb_services.file_path(kb["id"], doc["id"]).exists()

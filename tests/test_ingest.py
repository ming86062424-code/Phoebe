import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from app.config import Settings
from app.db import connect, init_db, now_iso
from app.kb import store
from app.kb.parsers.base import ParsedChunk, ParseError
from app.kb.services import build_kb_services
from tests.fakes import FakeParser
from tests.fakes_embedder import FakeEmbedder
from tests.fakes_store import InMemoryChunkStore


@pytest.fixture
def env(tmp_path):
    settings = Settings(
        db_path=str(tmp_path / "t.db"),
        kb_dir=str(tmp_path / "kb"),
    )
    conn = connect(settings.db_path)
    init_db(conn)
    conn.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES ('u', 'h', ?)", (now_iso(),)
    )
    conn.commit()
    kb = store.create_kb(conn, 1, "k", "")
    parser = FakeParser()
    services = build_kb_services(
        settings, embedder=FakeEmbedder(),
        parsers=[parser],
        threaded=False,
        store=InMemoryChunkStore(),
    )

    def add_doc():
        doc = store.create_document(conn, kb["id"], "a.pdf", "fake", 10)
        path = services.file_path(kb["id"], doc["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF")
        return doc

    yield conn, kb, parser, services, add_doc
    conn.close()


def test_process_marks_ready_and_indexes(env):
    conn, kb, parser, services, add_doc = env
    doc = add_doc()
    services.ingester.enqueue(doc["id"])
    got = store.get_document(conn, doc["id"])
    assert (got["status"], got["chunk_count"], got["progress_done"]) == ("ready", 2, 2)
    assert services.index.search(kb["id"], "cherry", 1)[0].page == 2
    assert parser.released == 1


def test_store_down_marks_failed_with_clear_reason(env, monkeypatch):
    from app.kb.chunk_store import KnowledgeStoreError

    conn, _, _, services, add_doc = env

    def boom(*args, **kwargs):
        raise KnowledgeStoreError("無法連線到 Milvus")

    monkeypatch.setattr(services.index._store, "insert", boom)
    doc = add_doc()
    services.ingester.enqueue(doc["id"])
    got = store.get_document(conn, doc["id"])
    assert got["status"] == "failed" and "無法連線到 Milvus" in got["error"]


def test_parse_error_marks_failed(env):
    conn, _, parser, services, add_doc = env
    parser.error = ParseError("壞掉了")
    doc = add_doc()
    services.ingester.enqueue(doc["id"])
    got = store.get_document(conn, doc["id"])
    assert (got["status"], got["error"]) == ("failed", "壞掉了")


def test_no_text_marks_failed(env):
    conn, _, parser, services, add_doc = env
    parser.pages = [ParsedChunk(1, "")]
    doc = add_doc()
    services.ingester.enqueue(doc["id"])
    assert "LightOnOCR" in store.get_document(conn, doc["id"])["error"]


def test_unknown_parser_marks_failed(env):
    conn, kb, _, services, _ = env
    doc = store.create_document(conn, kb["id"], "a.pdf", "nope", 1)
    services.ingester.enqueue(doc["id"])
    assert store.get_document(conn, doc["id"])["status"] == "failed"


def test_start_requeues_stale_documents(env):
    conn, _, parser, services, add_doc = env
    doc = add_doc()
    store.update_document(conn, doc["id"], status="indexing", progress_done=1)
    conn.execute("UPDATE kb_documents SET updated_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (doc["id"],))
    conn.commit()
    services.ingester.start()
    assert store.get_document(conn, doc["id"])["status"] == "ready"
    assert len(parser.parsed) == 1


def test_threaded_ingester_processes_queue(env):
    conn, _, _, services, add_doc = env
    services.ingester.threaded = True
    services.ingester.start()
    try:
        doc = add_doc()
        services.ingester.enqueue(doc["id"])
        deadline = time.time() + 10
        while store.get_document(conn, doc["id"])["status"] != "ready" and time.time() < deadline:
            time.sleep(0.05)
        assert store.get_document(conn, doc["id"])["status"] == "ready"
    finally:
        services.ingester.stop()


def test_document_replaced_mid_parse_is_not_indexed_as_the_new_one(env):
    """A deleted document whose id is reused must not be indexed or marked ready."""
    conn, kb, parser, services, add_doc = env
    doc = add_doc()

    original_parse = parser.parse

    def parse_then_replace(path, out_dir, on_progress):
        pages = original_parse(path, out_dir, on_progress)
        other = connect(services.ingester.db_path)
        other.execute("DELETE FROM kb_documents WHERE id = ?", (doc["id"],))
        other.execute(
            "INSERT INTO kb_documents (id, kb_id, filename, parser, status, created_at, updated_at) "
            "VALUES (?, ?, 'b.pdf', 'fake', 'queued', 'later', 'later')",
            (doc["id"], kb["id"]),
        )
        other.commit()
        other.close()
        return pages

    parser.parse = parse_then_replace
    assert store.claim_next_document(conn) == doc["id"]
    services.ingester.process(doc["id"])
    got = store.get_document(conn, doc["id"])
    assert (got["filename"], got["status"]) == ("b.pdf", "queued")
    assert services.index.search(kb["id"], "apple", 5) == []


def test_deleting_a_document_stops_parsing_at_the_next_page(env):
    conn, kb, parser, services, add_doc = env
    doc = add_doc()
    pages_done = []

    def slow_parse(path, out_dir, on_progress):
        for page in (1, 2, 3):
            if page == 2:
                store.delete_document(conn, kb["id"], doc["id"])
            on_progress(page, 3)
            pages_done.append(page)
        return [ParsedChunk(1, "apple")]

    parser.parse = slow_parse
    assert store.claim_next_document(conn) == doc["id"]
    services.ingester.process(doc["id"])
    assert pages_done == [1]
    assert services.index.search(kb["id"], "apple", 5) == []


def test_start_leaves_fresh_documents_alone(env):
    conn, _, parser, services, add_doc = env
    doc = add_doc()
    store.update_document(conn, doc["id"], status="parsing")
    services.ingester.start()
    assert store.get_document(conn, doc["id"])["status"] == "parsing"
    assert parser.parsed == []


def test_enqueue_unthreaded_processes_claimed_documents(env):
    conn, _, parser, services, add_doc = env
    first, second = add_doc(), add_doc()
    services.ingester.enqueue(first["id"])
    assert [store.get_document(conn, d["id"])["status"] for d in (first, second)] == ["ready", "ready"]
    assert len(parser.parsed) == 2


def test_run_forever_stops_on_event(env):
    _, _, _, services, _ = env
    stop = threading.Event()
    stop.set()
    services.ingester.run_forever(stop)  # returns instead of blocking


def test_run_forever_processes_then_stops_promptly(env):
    conn, _, _, services, add_doc = env
    services.ingester.poll_seconds = 30
    doc = add_doc()
    stop = threading.Event()
    t = threading.Thread(target=services.ingester.run_forever, args=(stop,))
    t.start()
    deadline = time.time() + 10
    while store.get_document(conn, doc["id"])["status"] != "ready" and time.time() < deadline:
        time.sleep(0.05)
    stop.set()
    services.ingester.wake()
    t.join(timeout=5)
    assert not t.is_alive()
    assert store.get_document(conn, doc["id"])["status"] == "ready"


def test_cleanup_orphans_removes_files_without_documents(env):
    conn, kb, _, services, add_doc = env
    keep = add_doc()
    keep_path = services.file_path(kb["id"], keep["id"])
    orphan = services.file_path(kb["id"], 9999)
    orphan.write_bytes(b"%PDF")
    tmp = services.kb_dir / str(kb["id"]) / "upload-abc.tmp"
    tmp.write_bytes(b"x")
    gone_kb = services.kb_dir / "777"
    gone_kb.mkdir()
    (gone_kb / "1.pdf").write_bytes(b"%PDF")
    assert services.ingester.cleanup_orphans() == 2
    assert keep_path.exists() and tmp.exists()
    assert not orphan.exists() and not gone_kb.exists()


def _age(conn, doc_id, ts="2000-01-01T00:00:00+00:00"):
    conn.execute("UPDATE kb_documents SET updated_at = ? WHERE id = ?", (ts, doc_id))
    conn.commit()


def test_heartbeat_refreshes_updated_at_during_slow_parse(env):
    conn, _, parser, services, add_doc = env
    services.ingester.heartbeat_seconds = 0.05
    doc = add_doc()
    seen = []
    original = parser.parse

    def slow_parse(path, out_dir, on_progress):
        _age(conn, doc["id"])  # as if no update for a long time
        time.sleep(0.4)  # no progress calls: only the heartbeat can refresh
        seen.append(store.get_document(conn, doc["id"])["updated_at"])
        return original(path, out_dir, on_progress)

    parser.parse = slow_parse
    assert store.claim_next_document(conn) == doc["id"]
    services.ingester.process(doc["id"])
    assert seen[0] > "2001"


def test_running_ingester_requeues_stale_documents_periodically(env, monkeypatch):
    import app.kb.ingest as ingest_mod

    conn, _, parser, services, add_doc = env
    monkeypatch.setattr(ingest_mod, "CLEANUP_INTERVAL_SECONDS", 0)
    services.ingester.poll_seconds = 0.05
    doc = add_doc()
    store.update_document(conn, doc["id"], status="parsing")
    _age(conn, doc["id"])
    stop = threading.Event()
    t = threading.Thread(target=services.ingester.run_forever, args=(stop,))
    t.start()
    deadline = time.time() + 10
    while store.get_document(conn, doc["id"])["status"] != "ready" and time.time() < deadline:
        time.sleep(0.05)
    stop.set()
    services.ingester.wake()
    t.join(timeout=5)
    assert store.get_document(conn, doc["id"])["status"] == "ready"


def test_active_document_is_not_requeued_by_another_ingester(env):
    conn, _, parser, services, add_doc = env
    services.ingester.heartbeat_seconds = 0.05
    doc = add_doc()
    original = parser.parse
    outcome = []

    def slow_parse(path, out_dir, on_progress):
        _age(conn, doc["id"])
        time.sleep(0.4)
        other = connect(services.ingester.db_path)
        try:
            stale_before = (
                datetime.now(timezone.utc) - timedelta(seconds=30)
            ).isoformat()
            outcome.append(store.requeue_stale_documents(other, stale_before))
        finally:
            other.close()
        return original(path, out_dir, on_progress)

    parser.parse = slow_parse
    assert store.claim_next_document(conn) == doc["id"]
    services.ingester.process(doc["id"])
    assert outcome == [[]]
    assert store.get_document(conn, doc["id"])["status"] == "ready"


def test_stop_event_ends_processing_after_current_page(env):
    conn, _, parser, services, add_doc = env
    doc = add_doc()
    stop = threading.Event()
    pages = []

    def parse(path, out_dir, on_progress):
        for page in (1, 2, 3):
            if page == 2:
                stop.set()
            on_progress(page, 3)
            pages.append(page)
        return [ParsedChunk(1, "x")]

    parser.parse = parse
    assert store.claim_next_document(conn) == doc["id"]
    services.ingester.process(doc["id"], stop)
    assert pages == [1]
    assert store.get_document(conn, doc["id"])["status"] == "parsing"


def test_pending_document_is_not_claimed_nor_orphaned(env):
    conn, kb, parser, services, add_doc = env
    doc = add_doc()
    store.update_document(conn, doc["id"], status="pending")
    assert store.claim_next_document(conn) is None
    assert services.ingester.cleanup_orphans() == 0
    assert services.file_path(kb["id"], doc["id"]).exists()


def test_stop_keeps_thread_reference_when_join_times_out(env):
    _, _, _, services, _ = env
    release = threading.Event()
    t = threading.Thread(target=release.wait)
    t.start()
    services.ingester._thread = t
    t.join = lambda timeout=None: None  # simulate a loop stuck in a long page
    services.ingester.stop()
    assert services.ingester._thread is t
    release.set()


# --- parser status published by the ingester, read by the web process ---


def _external_services(tmp_path, parser):
    settings = Settings(
        db_path=str(tmp_path / "x.db"),
        kb_dir=str(tmp_path / "kb"),
        kb_ingest_mode="external",
    )
    conn = connect(settings.db_path)
    init_db(conn)
    services = build_kb_services(
        settings, embedder=FakeEmbedder(),
        parsers=[parser],
        threaded=False,
        store=InMemoryChunkStore(),
    )
    return settings, conn, services


def test_external_availability_reads_parser_status(tmp_path):
    parser = FakeParser(available=False)  # the web side's own check must not matter
    _, conn, services = _external_services(tmp_path, parser)
    conn.execute(
        "INSERT INTO parser_status (parser_id, available, reason, updated_at) VALUES (?, 1, '', ?)",
        ("fake", now_iso()),
    )
    conn.commit()
    assert services.availability("fake") == (True, "")
    conn.execute("UPDATE parser_status SET available = 0, reason = '沒有 GPU'")
    conn.commit()
    assert services.availability("fake") == (False, "沒有 GPU")
    assert services.parser_info()[0]["available"] is False


def test_external_availability_without_row_or_when_stale(tmp_path):
    _, conn, services = _external_services(tmp_path, FakeParser())
    down = (False, "文件處理服務（ingest）未啟動")
    assert services.availability("fake") == down
    old = (datetime.now(timezone.utc) - timedelta(seconds=46)).isoformat()
    conn.execute(
        "INSERT INTO parser_status (parser_id, available, reason, updated_at) VALUES ('fake', 1, '', ?)",
        (old,),
    )
    conn.commit()
    assert services.availability("fake") == down


def test_embedded_availability_checks_locally(env):
    _, _, parser, services, _ = env
    assert services.availability("fake") == (True, "")
    parser.available = False
    assert services.availability("fake") == (False, "未安裝")


def test_ingester_publishes_parser_status(tmp_path):
    parser = FakeParser(available=False)
    _, conn, services = _external_services(tmp_path, parser)
    services.ingester.publish_parser_status()
    row = conn.execute("SELECT * FROM parser_status WHERE parser_id = 'fake'").fetchone()
    assert (row["available"], row["reason"]) == (0, "未安裝")
    parser.available = True
    services.ingester.publish_parser_status()
    row = conn.execute("SELECT * FROM parser_status WHERE parser_id = 'fake'").fetchone()
    assert (row["available"], row["reason"]) == (1, "")
    assert services.availability("fake") == (True, "")


def test_publisher_thread_publishes_and_stops(tmp_path):
    parser = FakeParser()
    _, conn, services = _external_services(tmp_path, parser)
    ingester = services.ingester
    ingester.start_status_publisher(interval=0.05)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if conn.execute("SELECT 1 FROM parser_status").fetchone():
                break
            time.sleep(0.02)
        else:
            raise AssertionError("status never published")
    finally:
        ingester.stop_status_publisher()
    assert ingester._status_thread is None


def test_cleanup_orphans_removes_old_upload_temp_files_only(env):
    import os

    _, kb, _, services, _ = env
    folder = services.kb_dir / str(kb["id"])
    folder.mkdir(parents=True, exist_ok=True)
    old = folder / "upload-old.tmp"
    fresh = folder / "upload-fresh.tmp"
    other = folder / "notes.tmp"
    for f in (old, fresh, other):
        f.write_bytes(b"x")
    past = time.time() - 2 * 3600
    os.utime(old, (past, past))
    os.utime(other, (past, past))
    assert services.ingester.cleanup_orphans() == 1
    assert not old.exists()
    assert fresh.exists() and other.exists()


def test_stale_pending_documents_are_failed(env):
    conn, _, _, services, add_doc = env
    stale = add_doc()
    fresh = add_doc()
    queued = add_doc()
    for d in (stale, fresh):
        store.update_document(conn, d["id"], status="pending")
    _age(conn, stale["id"])
    _age(conn, queued["id"])
    services.ingester.requeue_stale()
    got = store.get_document(conn, stale["id"])
    assert (got["status"], got["error"]) == ("failed", "上傳未完成，請重新上傳")
    assert store.get_document(conn, fresh["id"])["status"] == "pending"
    assert store.get_document(conn, queued["id"])["status"] == "queued"  # other statuses untouched


def test_parse_gets_a_fresh_document_folder_and_images_are_indexed(env):
    from app.kb.parsers.base import ImageRef

    conn, kb, parser, services, add_doc = env
    doc = add_doc()
    folder = services.doc_dir(kb["id"], doc["id"])
    folder.mkdir(parents=True)
    (folder / "stale.png").write_bytes(b"old attempt")

    def parse(path, out_dir, on_progress):
        assert out_dir == folder and out_dir.is_dir() and not (out_dir / "stale.png").exists()
        (out_dir / "page_1.png").write_bytes(b"png")
        return [ParsedChunk(1, "page text", page_image=ImageRef("page_1.png", 2, 2), kind="page_image")]

    parser.parse = parse
    services.ingester.enqueue(doc["id"])
    assert store.get_document(conn, doc["id"])["status"] == "ready"
    (row,) = services.index._store.rows.values()
    assert row.kind == "page_image" and row.page_image == ImageRef("page_1.png", 2, 2)


def test_cleanup_orphans_handles_new_file_types_and_document_folders(env):
    conn, kb, _, services, add_doc = env
    keep = add_doc()
    keep_dir = services.doc_dir(kb["id"], keep["id"])
    keep_dir.mkdir(parents=True)
    folder = services.kb_dir / str(kb["id"])
    orphans = [folder / "9001.txt", folder / "9002.png", folder / "9003.md"]
    for f in orphans:
        f.write_bytes(b"x")
    orphan_dir = folder / "9004"
    orphan_dir.mkdir()
    (orphan_dir / "page_1.png").write_bytes(b"x")
    assert services.ingester.cleanup_orphans() == 4
    assert keep_dir.exists() and services.file_path(kb["id"], keep["id"]).exists()
    assert not any(f.exists() for f in orphans) and not orphan_dir.exists()


def test_empty_text_file_fails_with_a_text_message(env):
    conn, kb, _, services, _ = env
    doc = store.create_document(conn, kb["id"], "a.txt", "text", 1)
    path = services.file_path(kb["id"], doc["id"], "a.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"  \n\n ")
    services.ingester.enqueue(doc["id"])
    got = store.get_document(conn, doc["id"])
    assert (got["status"], got["error"]) == ("failed", "檔案中沒有任何文字")


def test_superseded_parse_removes_the_document_folder(env):
    conn, kb, parser, services, add_doc = env
    doc = add_doc()
    folder = services.doc_dir(kb["id"], doc["id"])

    def parse(path, out_dir, on_progress):
        (out_dir / "page_1.png").write_bytes(b"png")
        other = connect(services.ingester.db_path)
        other.execute("DELETE FROM kb_documents WHERE id = ?", (doc["id"],))
        other.commit()
        other.close()
        on_progress(1, 1)
        return [ParsedChunk(1, "x")]

    parser.parse = parse
    assert store.claim_next_document(conn) == doc["id"]
    services.ingester.process(doc["id"])
    assert not folder.exists()
    assert (services.kb_dir / str(kb["id"])).exists()  # the knowledge base still exists


def test_superseded_parse_removes_the_folder_of_a_deleted_kb(env):
    conn, kb, parser, services, add_doc = env
    doc = add_doc()
    kb_folder = services.kb_dir / str(kb["id"])

    def parse(path, out_dir, on_progress):
        (out_dir / "page_1.png").write_bytes(b"png")
        other = connect(services.ingester.db_path)
        other.execute("DELETE FROM kb_documents WHERE kb_id = ?", (kb["id"],))
        other.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb["id"],))
        other.commit()
        other.close()
        on_progress(1, 1)
        return [ParsedChunk(1, "x")]

    parser.parse = parse
    assert store.claim_next_document(conn) == doc["id"]
    services.ingester.process(doc["id"])
    assert not kb_folder.exists()


def test_stale_folder_that_cannot_be_removed_fails_the_attempt(env, monkeypatch):
    from app.kb import ingest as ingest_module

    conn, kb, parser, services, add_doc = env
    doc = add_doc()
    folder = services.doc_dir(kb["id"], doc["id"])
    folder.mkdir(parents=True)
    (folder / "page_1.png").write_bytes(b"old")
    real = ingest_module._remove
    monkeypatch.setattr(ingest_module, "_remove", lambda p: False if p == folder else real(p))
    services.ingester.enqueue(doc["id"])
    got = store.get_document(conn, doc["id"])
    assert got["status"] == "failed" and "上次處理" in got["error"]
    assert parser.parsed == []


# ---- documents wait while their parser is unavailable (e.g. vLLM still loading) ----


def _two_parser_env(tmp_path, first_available=False):
    settings = Settings(db_path=str(tmp_path / "t.db"), kb_dir=str(tmp_path / "kb"))
    conn = connect(settings.db_path)
    init_db(conn)
    conn.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES ('u', 'h', ?)", (now_iso(),)
    )
    conn.commit()
    kb = store.create_kb(conn, 1, "k", "")
    ocr = FakeParser(available=first_available)
    ocr.id = "ocr"
    text = FakeParser()
    services = build_kb_services(
        settings, embedder=FakeEmbedder(), parsers=[ocr, text], threaded=False,
        store=InMemoryChunkStore(),
    )

    def add_doc(parser_id):
        doc = store.create_document(conn, kb["id"], "a.pdf", parser_id, 10)
        path = services.file_path(kb["id"], doc["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF")
        return doc

    return conn, ocr, text, services, add_doc


def test_document_waits_while_its_parser_is_unavailable(env):
    conn, _, parser, services, add_doc = env
    parser.available = False
    doc = add_doc()
    services.ingester.enqueue(doc["id"])
    got = store.get_document(conn, doc["id"])
    assert (got["status"], got["error"]) == ("queued", None)
    assert parser.parsed == []

    parser.available = True  # the OCR service finished loading
    services.ingester.enqueue(doc["id"])
    assert store.get_document(conn, doc["id"])["status"] == "ready"


def test_unavailable_parser_does_not_hold_up_other_documents(tmp_path):
    conn, ocr, text, services, add_doc = _two_parser_env(tmp_path)
    waiting, other = add_doc("ocr"), add_doc("fake")  # the waiting one is older
    services.ingester.enqueue(other["id"])
    assert store.get_document(conn, waiting["id"])["status"] == "queued"
    assert store.get_document(conn, other["id"])["status"] == "ready"
    assert ocr.parsed == [] and len(text.parsed) == 1
    conn.close()


def test_availability_is_checked_only_for_parsers_with_queued_documents(tmp_path, monkeypatch):
    conn, ocr, text, services, add_doc = _two_parser_env(tmp_path, first_available=True)
    checked: list[str] = []
    for p in (ocr, text):
        real = p.availability
        monkeypatch.setattr(p, "availability", lambda p=p, real=real: checked.append(p.id) or real())
    services.ingester.enqueue(0)  # nothing queued: no service is asked
    assert checked == []
    add_doc("fake")
    services.ingester.enqueue(0)
    assert checked == ["fake"]
    conn.close()


def test_failing_availability_check_counts_as_unavailable(env, monkeypatch):
    conn, _, parser, services, add_doc = env

    def broken():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(parser, "availability", broken)
    doc = add_doc()
    services.ingester.enqueue(doc["id"])
    assert store.get_document(conn, doc["id"])["status"] == "queued"


def test_unknown_parser_is_still_claimed_and_failed(env):
    conn, kb, _, services, _ = env
    doc = store.create_document(conn, kb["id"], "a.pdf", "gone", 10)
    services.ingester.enqueue(doc["id"])
    assert store.get_document(conn, doc["id"])["status"] == "failed"


# ---- several documents at once (INGEST_CONCURRENCY) ----


def _concurrent_env(tmp_path, concurrency, parsers=None):
    settings = Settings(
        db_path=str(tmp_path / "c.db"),
        kb_dir=str(tmp_path / "kb"),
        kb_ingest_poll_seconds=0.05,
        ingest_concurrency=concurrency,
    )
    conn = connect(settings.db_path)
    init_db(conn)
    conn.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES ('u', 'h', ?)", (now_iso(),)
    )
    conn.commit()
    kb = store.create_kb(conn, 1, "k", "")
    services = build_kb_services(
        settings, embedder=FakeEmbedder(), parsers=parsers or [FakeParser()], threaded=True,
        store=InMemoryChunkStore(),
    )

    def add_doc(parser_id="fake"):
        doc = store.create_document(conn, kb["id"], "a.pdf", parser_id, 10)
        path = services.file_path(kb["id"], doc["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF")
        return doc

    return conn, kb, services, add_doc


class _Running:
    """run_forever in a thread; stopped (with a bounded wait) when the block ends."""

    def __init__(self, ingester):
        self.ingester = ingester
        self.stop = threading.Event()
        self.thread = threading.Thread(target=ingester.run_forever, args=(self.stop,), daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.ingester.wake()
        self.thread.join(timeout=10)
        assert not self.thread.is_alive(), "run_forever did not return"


def _doc_id(path) -> int:
    from pathlib import Path

    return int(Path(path).stem)


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _status(conn, doc):
    got = store.get_document(conn, doc["id"])
    return None if got is None else got["status"]


def _all_settled(conn, docs):
    return lambda: all(_status(conn, d) in ("ready", "failed", None) for d in docs)


def test_concurrency_setting_reaches_the_ingester(tmp_path):
    conn, _, services, _ = _concurrent_env(tmp_path, 3)
    assert services.ingester.concurrency == 3
    conn.close()


def test_two_documents_are_processed_at_the_same_time(tmp_path):
    conn, _, services, add_doc = _concurrent_env(tmp_path, 2)
    parser = services.parsers["fake"]
    barrier = threading.Barrier(2, timeout=5)
    original = parser.parse

    def parse(path, out_dir, on_progress):
        barrier.wait()  # both must be inside parse at once
        return original(path, out_dir, on_progress)

    parser.parse = parse
    docs = [add_doc(), add_doc()]
    with _Running(services.ingester):
        assert _wait_for(_all_settled(conn, docs))
    assert [_status(conn, d) for d in docs] == ["ready", "ready"]
    conn.close()


def test_never_more_than_the_concurrency_limit(tmp_path):
    conn, _, services, add_doc = _concurrent_env(tmp_path, 2)
    parser = services.parsers["fake"]
    lock = threading.Lock()
    current = [0]
    peak = [0]
    original = parser.parse

    def parse(path, out_dir, on_progress):
        with lock:
            current[0] += 1
            peak[0] = max(peak[0], current[0])
        try:
            time.sleep(0.2)
            return original(path, out_dir, on_progress)
        finally:
            with lock:
                current[0] -= 1

    parser.parse = parse
    docs = [add_doc() for _ in range(5)]
    with _Running(services.ingester):
        assert _wait_for(_all_settled(conn, docs))
    assert [_status(conn, d) for d in docs] == ["ready"] * 5
    assert peak[0] == 2
    conn.close()


def test_concurrency_one_processes_in_order(tmp_path):
    conn, _, services, add_doc = _concurrent_env(tmp_path, 1)
    parser = services.parsers["fake"]
    order = []
    original = parser.parse

    def parse(path, out_dir, on_progress):
        order.append(_doc_id(path))
        return original(path, out_dir, on_progress)

    parser.parse = parse
    docs = [add_doc() for _ in range(3)]
    with _Running(services.ingester):
        assert _wait_for(_all_settled(conn, docs))
    assert order == [d["id"] for d in docs]
    conn.close()


def test_a_crashing_document_frees_its_slot(tmp_path, monkeypatch):
    conn, _, services, add_doc = _concurrent_env(tmp_path, 1)
    first, second = add_doc(), add_doc()
    ingester = services.ingester
    real = ingester.process

    def process(doc_id, stop=None):
        if doc_id == first["id"]:
            raise RuntimeError("boom")
        return real(doc_id, stop)

    monkeypatch.setattr(ingester, "process", process)
    with _Running(ingester):
        assert _wait_for(lambda: _status(conn, second) == "ready")
    assert _status(conn, first) == "parsing"  # left for the stale requeue
    conn.close()


def test_deleting_one_document_leaves_the_parallel_one_alone(tmp_path):
    conn, kb, services, add_doc = _concurrent_env(tmp_path, 2)
    parser = services.parsers["fake"]
    doomed, kept = add_doc(), add_doc()
    both_started = threading.Barrier(2, timeout=5)
    pages_done: dict[int, list[int]] = {doomed["id"]: [], kept["id"]: []}

    def parse(path, out_dir, on_progress):
        doc_id = _doc_id(path)
        for page in (1, 2, 3):
            on_progress(page, 3)
            pages_done[doc_id].append(page)
            if page == 1:
                both_started.wait()
                if doc_id == doomed["id"]:
                    other = connect(services.ingester.db_path)
                    store.delete_document(other, kb["id"], doomed["id"])
                    other.close()
        return [ParsedChunk(1, f"apple {doc_id}"), ParsedChunk(2, f"cherry {doc_id}")]

    parser.parse = parse
    with _Running(services.ingester):
        assert _wait_for(lambda: _status(conn, kept) == "ready")
    assert store.get_document(conn, doomed["id"]) is None
    assert pages_done[doomed["id"]] == [1]
    assert pages_done[kept["id"]] == [1, 2, 3]
    rows = list(services.index._store.rows.values())
    assert [r.document_id for r in rows] == [kept["id"], kept["id"]]
    conn.close()


def test_two_documents_of_one_knowledge_base_index_side_by_side(tmp_path):
    conn, kb, services, add_doc = _concurrent_env(tmp_path, 2)
    parser = services.parsers["fake"]
    barrier = threading.Barrier(2, timeout=5)

    def parse(path, out_dir, on_progress):
        barrier.wait()  # both index at about the same time
        on_progress(2, 2)
        doc_id = _doc_id(path)
        return [ParsedChunk(1, f"apple {doc_id}"), ParsedChunk(2, f"cherry {doc_id}")]

    parser.parse = parse
    docs = [add_doc(), add_doc()]
    with _Running(services.ingester):
        assert _wait_for(_all_settled(conn, docs))
    assert [_status(conn, d) for d in docs] == ["ready", "ready"]
    rows = list(services.index._store.rows.values())
    assert sorted(r.document_id for r in rows) == sorted([d["id"] for d in docs] * 2)
    assert all(r.kb_id == kb["id"] for r in rows)
    conn.close()


def test_unavailable_parser_still_waits_with_parallel_slots(tmp_path):
    ocr = FakeParser(available=False)
    ocr.id = "ocr"
    conn, _, services, add_doc = _concurrent_env(tmp_path, 2, parsers=[ocr, FakeParser()])
    waiting, other = add_doc("ocr"), add_doc("fake")  # the waiting one is older
    with _Running(services.ingester):
        assert _wait_for(lambda: _status(conn, other) == "ready")
        time.sleep(0.2)  # several more dispatcher rounds with free slots
        assert _status(conn, waiting) == "queued"
        assert ocr.parsed == []
        ocr.available = True  # the OCR service finished loading
        services.ingester.wake()
        assert _wait_for(lambda: _status(conn, waiting) == "ready")
    conn.close()


def test_parsers_are_released_only_when_all_slots_are_idle(tmp_path):
    conn, _, services, add_doc = _concurrent_env(tmp_path, 2)
    parser = services.parsers["fake"]
    slow, fast = add_doc(), add_doc()
    slow_started = threading.Event()
    finish_slow = threading.Event()
    original = parser.parse

    def parse(path, out_dir, on_progress):
        if _doc_id(path) == slow["id"]:
            slow_started.set()
            assert finish_slow.wait(5)
        return original(path, out_dir, on_progress)

    parser.parse = parse
    with _Running(services.ingester):
        assert slow_started.wait(5)
        assert _wait_for(lambda: _status(conn, fast) == "ready")
        time.sleep(0.2)  # several idle dispatcher rounds while the slow one still parses
        assert parser.released == 0
        finish_slow.set()
        assert _wait_for(lambda: _status(conn, slow) == "ready")
        assert _wait_for(lambda: parser.released >= 1)
    conn.close()


def test_stop_waits_for_running_documents(tmp_path):
    conn, _, services, add_doc = _concurrent_env(tmp_path, 2)
    parser = services.parsers["fake"]
    on_page_one = threading.Event()
    go_on = threading.Event()
    pages = []

    def parse(path, out_dir, on_progress):
        for page in (1, 2, 3):
            on_progress(page, 3)
            pages.append(page)
            if page == 1:
                on_page_one.set()
                assert go_on.wait(5)
        return [ParsedChunk(1, "x")]

    parser.parse = parse
    doc = add_doc()
    running = _Running(services.ingester)
    with running:
        assert on_page_one.wait(5)
        running.stop.set()
        services.ingester.wake()
        time.sleep(0.2)
        assert running.thread.is_alive()  # still waiting for the document in flight
        go_on.set()
        running.thread.join(timeout=10)
        assert not running.thread.is_alive()
    assert pages == [1]
    assert _status(conn, doc) == "parsing"  # the stale requeue picks it up later
    conn.close()

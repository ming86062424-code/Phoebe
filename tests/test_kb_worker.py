import time

import pytest

from app.config import Settings
from app.db import connect, init_db, now_iso
from app.kb import store
from app.kb.parsers.base import PageText, ParseError
from app.kb.services import build_kb_services
from tests.fakes import FakeParser, KeywordEmbedding


@pytest.fixture
def env(tmp_path):
    settings = Settings(
        db_path=str(tmp_path / "t.db"),
        kb_dir=str(tmp_path / "kb"),
        chroma_dir=str(tmp_path / "chroma"),
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
        settings, embed_model=KeywordEmbedding(), parsers=[parser], threaded=False
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
    services.worker.enqueue(doc["id"])
    got = store.get_document(conn, doc["id"])
    assert (got["status"], got["chunk_count"], got["progress_done"]) == ("ready", 2, 2)
    assert services.index.search(kb["id"], "cherry", 1)[0].page == 2
    assert parser.released == 1


def test_parse_error_marks_failed(env):
    conn, _, parser, services, add_doc = env
    parser.error = ParseError("壞掉了")
    doc = add_doc()
    services.worker.enqueue(doc["id"])
    got = store.get_document(conn, doc["id"])
    assert (got["status"], got["error"]) == ("failed", "壞掉了")


def test_no_text_marks_failed(env):
    conn, _, parser, services, add_doc = env
    parser.pages = [PageText(1, "")]
    doc = add_doc()
    services.worker.enqueue(doc["id"])
    assert "LightOnOCR" in store.get_document(conn, doc["id"])["error"]


def test_unknown_parser_marks_failed(env):
    conn, kb, _, services, _ = env
    doc = store.create_document(conn, kb["id"], "a.pdf", "nope", 1)
    services.worker.enqueue(doc["id"])
    assert store.get_document(conn, doc["id"])["status"] == "failed"


def test_start_requeues_unfinished_documents(env):
    conn, _, parser, services, add_doc = env
    doc = add_doc()
    store.update_document(conn, doc["id"], status="indexing", progress_done=1)
    services.worker.start()
    assert store.get_document(conn, doc["id"])["status"] == "ready"
    assert len(parser.parsed) == 1


def test_threaded_worker_processes_queue(env):
    conn, _, _, services, add_doc = env
    services.worker.threaded = True
    services.worker.start()
    try:
        doc = add_doc()
        services.worker.enqueue(doc["id"])
        deadline = time.time() + 10
        while store.get_document(conn, doc["id"])["status"] != "ready" and time.time() < deadline:
            time.sleep(0.05)
        assert store.get_document(conn, doc["id"])["status"] == "ready"
    finally:
        services.worker.stop()


def test_document_replaced_mid_parse_is_not_indexed_as_the_new_one(env):
    """A deleted document whose id is reused must not be indexed or marked ready."""
    conn, kb, parser, services, add_doc = env
    doc = add_doc()

    original_parse = parser.parse

    def parse_then_replace(path, on_progress):
        pages = original_parse(path, on_progress)
        other = connect(services.worker.db_path)
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
    services.worker.process(doc["id"])
    got = store.get_document(conn, doc["id"])
    assert (got["filename"], got["status"]) == ("b.pdf", "queued")
    assert services.index.search(kb["id"], "apple", 5) == []

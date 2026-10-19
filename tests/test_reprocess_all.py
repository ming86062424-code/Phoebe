import pytest

from app.db import connect, init_db, now_iso
from app.kb import store
from app.kb.reprocess_all import main, reprocess


@pytest.fixture
def conn(tmp_path):
    c = connect(str(tmp_path / "r.db"))
    init_db(c)
    c.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES ('alice', 'h', ?)",
        (now_iso(),),
    )
    c.commit()
    return c


def _doc(conn, kb_id, filename, status, parser="liteparse", **fields):
    doc = store.create_document(conn, kb_id, filename, parser, 10, status=status)
    if fields:
        store.update_document(conn, doc["id"], **fields)
    return doc["id"]


def test_ready_and_failed_documents_are_requeued_with_clean_state(conn):
    kb = store.create_kb(conn, 1, "手冊", "")["id"]
    ready = _doc(conn, kb, "a.pdf", "ready", chunk_count=7, progress_done=3, progress_total=3)
    failed = _doc(conn, kb, "b.pdf", "failed", parser="lightonocr", error="壞了")

    assert reprocess(conn, dry_run=False) == {kb: 2}

    a = store.get_document(conn, ready)
    assert (a["status"], a["error"], a["chunk_count"], a["progress_done"]) == ("queued", None, 0, 0)
    b = store.get_document(conn, failed)
    assert (b["status"], b["error"]) == ("queued", None)
    assert b["parser"] == "lightonocr"  # the original parser is kept


def test_documents_being_processed_are_skipped(conn):
    kb = store.create_kb(conn, 1, "手冊", "")["id"]
    busy = [_doc(conn, kb, f"{s}.pdf", s) for s in ("parsing", "indexing", "pending")]
    ready = _doc(conn, kb, "r.pdf", "ready")

    assert reprocess(conn, dry_run=False) == {kb: 1}

    assert [store.get_document(conn, i)["status"] for i in busy] == ["parsing", "indexing", "pending"]
    assert store.get_document(conn, ready)["status"] == "queued"


def test_already_queued_documents_stay_queued_and_are_not_counted_twice(conn):
    kb = store.create_kb(conn, 1, "手冊", "")["id"]
    queued = _doc(conn, kb, "q.pdf", "queued")
    assert reprocess(conn, dry_run=False) == {}
    assert store.get_document(conn, queued)["status"] == "queued"


def test_counts_are_per_knowledge_base(conn):
    kb1 = store.create_kb(conn, 1, "一", "")["id"]
    kb2 = store.create_kb(conn, 1, "二", "")["id"]
    kb3 = store.create_kb(conn, 1, "空", "")["id"]
    _doc(conn, kb1, "a.pdf", "ready")
    _doc(conn, kb1, "b.txt", "ready", parser="text")
    _doc(conn, kb2, "c.png", "failed", parser="image")

    assert reprocess(conn, dry_run=False) == {kb1: 2, kb2: 1}
    assert kb3 not in reprocess(conn, dry_run=True)


def test_dry_run_reports_counts_and_changes_nothing(conn):
    kb = store.create_kb(conn, 1, "手冊", "")["id"]
    ready = _doc(conn, kb, "a.pdf", "ready", chunk_count=5)
    before = store.get_document(conn, ready)

    assert reprocess(conn, dry_run=True) == {kb: 1}

    assert store.get_document(conn, ready) == before


def test_main_prints_counts_and_honours_dry_run(tmp_path, monkeypatch, capsys):
    db = str(tmp_path / "m.db")
    c = connect(db)
    init_db(c)
    c.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES ('alice', 'h', ?)",
        (now_iso(),),
    )
    c.commit()
    kb = store.create_kb(c, 1, "手冊", "")["id"]
    doc = _doc(c, kb, "a.pdf", "ready")
    c.close()
    monkeypatch.setenv("DB_PATH", db)

    assert main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"知識庫 {kb}：排入 1 份" in out and "試跑" in out
    c = connect(db)
    assert store.get_document(c, doc)["status"] == "ready"

    assert main([]) == 0
    assert store.get_document(c, doc)["status"] == "queued"
    assert "完成：共 1 份" in capsys.readouterr().out
    c.close()

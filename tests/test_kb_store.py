import pytest

from app.db import connect, init_db, now_iso
from app.kb import store


@pytest.fixture
def conn(tmp_path):
    c = connect(str(tmp_path / "t.db"))
    init_db(c)
    for name in ("alice", "bob"):
        c.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, 'h', ?)",
            (name, now_iso()),
        )
    c.commit()
    return c


def test_create_and_list_kbs_are_per_user(conn):
    kb = store.create_kb(conn, 1, "手冊", "說明")
    store.create_kb(conn, 2, "別人的", "")
    kbs = store.list_kbs(conn, 1)
    assert [k["name"] for k in kbs] == ["手冊"]
    assert kbs[0]["document_count"] == 0
    assert store.get_kb(conn, 2, kb["id"]) is None
    assert store.get_kb(conn, 1, kb["id"])["description"] == "說明"


def test_update_and_delete_kb(conn):
    kb = store.create_kb(conn, 1, "a", "")
    assert store.update_kb(conn, 1, kb["id"], name="b")["name"] == "b"
    assert store.update_kb(conn, 2, kb["id"], name="c") is None
    assert store.delete_kb(conn, 2, kb["id"]) is False
    assert store.delete_kb(conn, 1, kb["id"]) is True
    assert store.list_kbs(conn, 1) == []


def test_document_lifecycle(conn):
    kb = store.create_kb(conn, 1, "a", "")
    doc = store.create_document(conn, kb["id"], "x.pdf", "liteparse", 123)
    assert doc["status"] == "queued"
    store.update_document(conn, doc["id"], status="parsing", progress_done=1, progress_total=3)
    got = store.get_document(conn, doc["id"])
    assert (got["status"], got["progress_done"], got["progress_total"]) == ("parsing", 1, 3)
    assert [d["id"] for d in store.list_documents(conn, kb["id"])] == [doc["id"]]
    assert store.list_kbs(conn, 1)[0]["document_count"] == 1
    assert [d["id"] for d in store.list_unfinished_documents(conn)] == [doc["id"]]
    store.update_document(conn, doc["id"], status="ready")
    assert store.list_unfinished_documents(conn) == []
    assert store.delete_document(conn, kb["id"], doc["id"]) is True
    assert store.get_document(conn, doc["id"]) is None


def test_update_document_rejects_unknown_field(conn):
    kb = store.create_kb(conn, 1, "a", "")
    doc = store.create_document(conn, kb["id"], "x.pdf", "liteparse", 1)
    with pytest.raises(ValueError):
        store.update_document(conn, doc["id"], filename="evil")


def _queued_doc(conn, kb_id, name="a.pdf"):
    return store.create_document(conn, kb_id, name, "fake", 1)


def test_claim_next_document_claims_oldest_queued(conn):
    kb = store.create_kb(conn, 1, "a", "")
    first = _queued_doc(conn, kb["id"])
    second = _queued_doc(conn, kb["id"], "b.pdf")
    assert store.claim_next_document(conn) == first["id"]
    assert store.get_document(conn, first["id"])["status"] == "parsing"
    assert store.get_document(conn, second["id"])["status"] == "queued"
    assert store.claim_next_document(conn) == second["id"]
    assert store.claim_next_document(conn) is None


def test_two_connections_claim_distinct_documents(conn, tmp_path):
    kb = store.create_kb(conn, 1, "a", "")
    _queued_doc(conn, kb["id"])
    _queued_doc(conn, kb["id"], "b.pdf")
    other = connect(str(tmp_path / "t.db"))
    try:
        a = store.claim_next_document(conn)
        b = store.claim_next_document(other)
        assert a is not None and b is not None and a != b
        assert store.claim_next_document(conn) is None
        assert store.claim_next_document(other) is None
    finally:
        other.close()


def test_stale_parsing_documents_are_requeued(conn):
    kb = store.create_kb(conn, 1, "a", "")
    old = _queued_doc(conn, kb["id"])
    fresh = _queued_doc(conn, kb["id"], "b.pdf")
    store.update_document(conn, old["id"], status="parsing", progress_done=3)
    store.update_document(conn, fresh["id"], status="indexing")
    conn.execute("UPDATE kb_documents SET updated_at = '2000-01-01T00:00:00' WHERE id = ?", (old["id"],))
    conn.commit()
    assert store.requeue_stale_documents(conn, "2020-01-01T00:00:00") == [old["id"]]
    got = store.get_document(conn, old["id"])
    assert (got["status"], got["progress_done"]) == ("queued", 0)
    assert store.get_document(conn, fresh["id"])["status"] == "indexing"

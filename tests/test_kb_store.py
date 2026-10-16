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

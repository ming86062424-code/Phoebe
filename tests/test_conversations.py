from contextlib import contextmanager

from app import conversations
from app.db import connect


@contextmanager
def db(settings):
    conn = connect(settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


def test_create_and_list(make_user):
    alice = make_user("alice")
    r = alice.post("/api/conversations")
    assert r.status_code == 201
    items = alice.get("/api/conversations").json()
    assert len(items) == 1
    assert items[0]["title"] == ""
    assert set(items[0]) == {"id", "title", "created_at", "updated_at"}


def test_list_orders_by_updated_at_then_id_desc(make_user, settings):
    alice = make_user("alice")
    ids = [alice.post("/api/conversations").json()["id"] for _ in range(3)]
    c1, c2, c3 = ids
    assert [c["id"] for c in alice.get("/api/conversations").json()] == [c3, c2, c1]
    with db(settings) as conn:
        conversations.add_message(conn, c1, "user", "hi")
    assert [c["id"] for c in alice.get("/api/conversations").json()] == [c1, c3, c2]


def test_get_includes_messages_in_order(make_user, settings):
    alice = make_user("alice")
    cid = alice.post("/api/conversations").json()["id"]
    with db(settings) as conn:
        conversations.add_message(conn, cid, "user", "q")
        conversations.add_message(conn, cid, "assistant", "a")
        assert conversations.get_history(conn, cid) == [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    data = alice.get(f"/api/conversations/{cid}").json()
    assert [m["role"] for m in data["messages"]] == ["user", "assistant"]
    assert set(data["messages"][0]) == {"id", "role", "content", "created_at"}


def test_message_content_stored_verbatim(make_user, settings):
    alice = make_user("alice")
    cid = alice.post("/api/conversations").json()["id"]
    text = "  <script>alert(1)</script> "
    with db(settings) as conn:
        conversations.add_message(conn, cid, "user", text)
    data = alice.get(f"/api/conversations/{cid}").json()
    assert data["messages"][0]["content"] == text


def test_delete_removes_conversation_and_messages(make_user, settings):
    alice = make_user("alice")
    cid = alice.post("/api/conversations").json()["id"]
    with db(settings) as conn:
        conversations.add_message(conn, cid, "user", "x")
    assert alice.delete(f"/api/conversations/{cid}").status_code == 204
    assert alice.get(f"/api/conversations/{cid}").status_code == 404
    with db(settings) as conn:
        n = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert n == 0


def test_user_cannot_read_or_delete_others_conversation(make_user):
    alice, bob = make_user("alice"), make_user("bob")
    cid = alice.post("/api/conversations").json()["id"]
    assert bob.get(f"/api/conversations/{cid}").status_code == 404
    assert bob.delete(f"/api/conversations/{cid}").status_code == 404
    assert bob.get("/api/conversations").json() == []
    assert alice.get(f"/api/conversations/{cid}").status_code == 200


def test_requires_login(client):
    assert client.get("/api/conversations").status_code == 401
    assert client.post("/api/conversations").status_code == 401
    assert client.get("/api/conversations/1").status_code == 401
    assert client.delete("/api/conversations/1").status_code == 401


def test_set_title_if_empty_truncates_to_30_and_only_once(make_user, settings):
    alice = make_user("alice")
    cid = alice.post("/api/conversations").json()["id"]
    with db(settings) as conn:
        conversations.set_title_if_empty(conn, cid, "  " + "一" * 40)
        conversations.set_title_if_empty(conn, cid, "other")
    assert alice.get(f"/api/conversations/{cid}").json()["title"] == "一" * 30

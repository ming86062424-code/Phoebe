import json
import sqlite3

from app.db import now_iso


def create_conversation(conn: sqlite3.Connection, user_id: int) -> dict:
    now = now_iso()
    cur = conn.execute(
        "INSERT INTO conversations (user_id, title, created_at, updated_at) VALUES (?, '', ?, ?)",
        (user_id, now, now),
    )
    conn.commit()
    return {"id": cur.lastrowid, "title": "", "kb_id": None, "created_at": now, "updated_at": now}


def list_conversations(conn: sqlite3.Connection, user_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT id, title, kb_id, created_at, updated_at FROM conversations "
        "WHERE user_id = ? ORDER BY updated_at DESC, id DESC",
        (user_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(conn: sqlite3.Connection, user_id: int, conversation_id: int) -> dict | None:
    row = conn.execute(
        "SELECT id, title, kb_id, created_at, updated_at FROM conversations "
        "WHERE id = ? AND user_id = ?",
        (conversation_id, user_id),
    ).fetchone()
    if row is None:
        return None
    msgs = conn.execute(
        "SELECT id, role, content, sources, created_at FROM messages "
        "WHERE conversation_id = ? ORDER BY id",
        (conversation_id,),
    ).fetchall()
    messages = [
        {**dict(m), "sources": json.loads(m["sources"]) if m["sources"] else None} for m in msgs
    ]
    return {**dict(row), "messages": messages}


def set_kb(conn: sqlite3.Connection, conversation_id: int, kb_id: int | None) -> None:
    conn.execute("UPDATE conversations SET kb_id = ? WHERE id = ?", (kb_id, conversation_id))
    conn.commit()


def delete_conversation(conn: sqlite3.Connection, user_id: int, conversation_id: int) -> bool:
    cur = conn.execute(
        "DELETE FROM conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id)
    )
    conn.commit()
    return cur.rowcount > 0


def add_message(
    conn: sqlite3.Connection,
    conversation_id: int,
    role: str,
    content: str,
    sources: list[dict] | None = None,
) -> int:
    now = now_iso()
    cur = conn.execute(
        "INSERT INTO messages (conversation_id, role, content, sources, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            conversation_id,
            role,
            content,
            None if sources is None else json.dumps(sources, ensure_ascii=False),
            now,
        ),
    )
    conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
    conn.commit()
    return cur.lastrowid


def set_title_if_empty(conn: sqlite3.Connection, conversation_id: int, text: str) -> None:
    conn.execute(
        "UPDATE conversations SET title = ? WHERE id = ? AND title = ''",
        (text.strip()[:30], conversation_id),
    )
    conn.commit()


def get_history(conn: sqlite3.Connection, conversation_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
        (conversation_id,),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]

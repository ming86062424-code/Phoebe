import sqlite3

from app.db import now_iso

DOC_COLUMNS = (
    "id, kb_id, filename, parser, status, error, progress_done, progress_total, "
    "chunk_count, size_bytes, created_at, updated_at"
)
UPDATABLE_DOC_FIELDS = {"status", "error", "progress_done", "progress_total", "chunk_count"}
UNFINISHED_STATUSES = ("pending", "queued", "parsing", "indexing")


def create_kb(
    conn: sqlite3.Connection,
    user_id: int,
    name: str,
    description: str,
    parser: str = "liteparse",
) -> dict:
    now = now_iso()
    cur = conn.execute(
        "INSERT INTO knowledge_bases (user_id, name, description, parser, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, name, description, parser, now, now),
    )
    conn.commit()
    return get_kb(conn, user_id, cur.lastrowid)


def list_kbs(conn: sqlite3.Connection, user_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT k.id, k.name, k.description, k.parser, k.created_at, k.updated_at, "
        "COUNT(d.id) AS document_count "
        "FROM knowledge_bases k LEFT JOIN kb_documents d ON d.kb_id = k.id "
        "WHERE k.user_id = ? GROUP BY k.id ORDER BY k.updated_at DESC, k.id DESC",
        (user_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_kb(conn: sqlite3.Connection, user_id: int, kb_id: int) -> dict | None:
    row = conn.execute(
        "SELECT id, name, description, parser, created_at, updated_at FROM knowledge_bases "
        "WHERE id = ? AND user_id = ?",
        (kb_id, user_id),
    ).fetchone()
    return dict(row) if row else None


def update_kb(
    conn: sqlite3.Connection,
    user_id: int,
    kb_id: int,
    name: str | None = None,
    description: str | None = None,
    parser: str | None = None,
) -> dict | None:
    """Change the given fields; the parser only applies to documents uploaded later."""
    kb = get_kb(conn, user_id, kb_id)
    if kb is None:
        return None
    conn.execute(
        "UPDATE knowledge_bases SET name = ?, description = ?, parser = ?, updated_at = ? "
        "WHERE id = ?",
        (
            kb["name"] if name is None else name,
            kb["description"] if description is None else description,
            kb["parser"] if parser is None else parser,
            now_iso(),
            kb_id,
        ),
    )
    conn.commit()
    return get_kb(conn, user_id, kb_id)


def delete_kb(conn: sqlite3.Connection, user_id: int, kb_id: int) -> bool:
    cur = conn.execute(
        "DELETE FROM knowledge_bases WHERE id = ? AND user_id = ?", (kb_id, user_id)
    )
    conn.commit()
    return cur.rowcount > 0


def create_document(
    conn: sqlite3.Connection,
    kb_id: int,
    filename: str,
    parser: str,
    size_bytes: int,
    status: str = "queued",
) -> dict:
    now = now_iso()
    cur = conn.execute(
        "INSERT INTO kb_documents (kb_id, filename, parser, status, size_bytes, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (kb_id, filename, parser, status, size_bytes, now, now),
    )
    conn.execute("UPDATE knowledge_bases SET updated_at = ? WHERE id = ?", (now, kb_id))
    conn.commit()
    return get_document(conn, cur.lastrowid)


def get_document(conn: sqlite3.Connection, doc_id: int) -> dict | None:
    row = conn.execute(
        f"SELECT {DOC_COLUMNS} FROM kb_documents WHERE id = ?", (doc_id,)
    ).fetchone()
    return dict(row) if row else None


def list_documents(conn: sqlite3.Connection, kb_id: int) -> list[dict]:
    rows = conn.execute(
        f"SELECT {DOC_COLUMNS} FROM kb_documents WHERE kb_id = ? ORDER BY id DESC", (kb_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def list_unfinished_documents(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        f"SELECT {DOC_COLUMNS} FROM kb_documents WHERE status IN (?, ?, ?, ?) ORDER BY id",
        UNFINISHED_STATUSES,
    ).fetchall()
    return [dict(r) for r in rows]


def queued_parsers(conn: sqlite3.Connection) -> set[str]:
    """The parsers that queued documents are waiting for."""
    rows = conn.execute(
        "SELECT DISTINCT parser FROM kb_documents WHERE status = 'queued'"
    ).fetchall()
    return {r["parser"] for r in rows}


def claim_next_document(
    conn: sqlite3.Connection, skip_parsers: tuple[str, ...] = ()
) -> int | None:
    """Atomically take the oldest queued document (queued -> parsing); None if there is none.

    Documents of `skip_parsers` (parsers that cannot run right now) stay queued.
    """
    skip = f" AND parser NOT IN ({', '.join('?' * len(skip_parsers))})" if skip_parsers else ""
    row = conn.execute(
        "UPDATE kb_documents SET status = 'parsing', error = NULL, progress_done = 0, updated_at = ? "
        f"WHERE id = (SELECT id FROM kb_documents WHERE status = 'queued'{skip} ORDER BY id LIMIT 1) "
        "AND status = 'queued' RETURNING id",
        (now_iso(), *skip_parsers),
    ).fetchone()
    conn.commit()
    return row["id"] if row else None


def requeue_stale_documents(conn: sqlite3.Connection, stale_before: str) -> list[int]:
    """Put documents whose worker went away (no update since stale_before) back in the queue."""
    rows = conn.execute(
        "UPDATE kb_documents SET status = 'queued', progress_done = 0, updated_at = ? "
        "WHERE status IN ('parsing', 'indexing') AND updated_at < ? RETURNING id",
        (now_iso(), stale_before),
    ).fetchall()
    conn.commit()
    return sorted(r["id"] for r in rows)


def fail_stale_pending_documents(
    conn: sqlite3.Connection, stale_before: str, message: str
) -> list[int]:
    """Fail uploads that never finished (the web process died between 'pending' and 'queued')."""
    rows = conn.execute(
        "UPDATE kb_documents SET status = 'failed', error = ?, updated_at = ? "
        "WHERE status = 'pending' AND updated_at < ? RETURNING id",
        (message, now_iso(), stale_before),
    ).fetchall()
    conn.commit()
    return sorted(r["id"] for r in rows)


def touch_document(conn: sqlite3.Connection, doc_id: int) -> None:
    """Heartbeat: show that a worker is still busy with this document."""
    conn.execute(
        "UPDATE kb_documents SET updated_at = ? WHERE id = ? AND status IN ('parsing', 'indexing')",
        (now_iso(), doc_id),
    )
    conn.commit()


def update_document(conn: sqlite3.Connection, doc_id: int, **fields) -> None:
    unknown = set(fields) - UPDATABLE_DOC_FIELDS
    if unknown:
        raise ValueError(f"不可更新的欄位：{sorted(unknown)}")
    assignments = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(
        f"UPDATE kb_documents SET {assignments}, updated_at = ? WHERE id = ?",
        (*fields.values(), now_iso(), doc_id),
    )
    conn.commit()


def delete_document(conn: sqlite3.Connection, kb_id: int, doc_id: int) -> bool:
    cur = conn.execute(
        "DELETE FROM kb_documents WHERE id = ? AND kb_id = ?", (doc_id, kb_id)
    )
    conn.commit()
    return cur.rowcount > 0


def requeue_finished_documents(
    conn: sqlite3.Connection, dry_run: bool = False
) -> dict[int, int]:
    """Put every 'ready' or 'failed' document back in the queue; returns kb_id -> count.

    Documents a worker may be handling (pending, parsing, indexing) are left alone, and
    queued ones are already in line. The status check is part of the UPDATE, so a document
    that a worker claims in the meantime is not touched. error, chunk_count and progress
    are reset the same way a fresh upload starts.
    """
    if dry_run:
        rows = conn.execute(
            "SELECT kb_id, id FROM kb_documents WHERE status IN ('ready', 'failed')"
        ).fetchall()
    else:
        rows = conn.execute(
            "UPDATE kb_documents SET status = 'queued', error = NULL, chunk_count = 0, "
            "progress_done = 0, progress_total = 0, updated_at = ? "
            "WHERE status IN ('ready', 'failed') RETURNING kb_id, id",
            (now_iso(),),
        ).fetchall()
        conn.commit()
    counts: dict[int, int] = {}
    for row in rows:
        counts[row["kb_id"]] = counts.get(row["kb_id"], 0) + 1
    return counts

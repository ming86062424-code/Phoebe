import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS active_replies (
    conversation_id INTEGER PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    started_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS parser_status (
    parser_id TEXT PRIMARY KEY,
    available INTEGER NOT NULL,
    reason TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

# Knowledge base ids name files and filter Milvus chunks (kb_id), so they must never be reused
# after a delete: these tables use AUTOINCREMENT.
AUTOINCREMENT_TABLES = {
    "knowledge_bases": """CREATE TABLE {name} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)""",
    "kb_documents": """CREATE TABLE {name} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kb_id INTEGER NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    parser TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'queued', 'parsing', 'indexing', 'ready', 'failed')),
    error TEXT,
    progress_done INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)""",
}

# Columns added after the first release; init_db adds them to older databases.
MIGRATIONS = [
    (
        "conversations",
        "kb_id",
        "INTEGER REFERENCES knowledge_bases(id) ON DELETE SET NULL",
    ),
    ("messages", "sources", "TEXT"),
]


def _enable_wal(conn: sqlite3.Connection) -> None:
    # Converting a fresh database to WAL can fail with "database is locked" without
    # consulting busy_timeout when several processes connect at once: retry.
    deadline = time.monotonic() + 10
    while True:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # busy_timeout first so concurrent first-time WAL conversion waits instead of failing
    conn.execute("PRAGMA busy_timeout=5000")
    _enable_wal(conn)
    return conn


def _table_sql(conn: sqlite3.Connection, name: str) -> str | None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row["sql"] if row else None


def _ensure_autoincrement(conn: sqlite3.Connection) -> None:
    """Create the AUTOINCREMENT tables, rebuilding ones made by older versions.

    Must run inside the init_db transaction (no commits here).
    """
    for name, ddl in AUTOINCREMENT_TABLES.items():
        sql = _table_sql(conn, name)
        if sql is None:
            conn.execute(ddl.format(name=name).replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ", 1))
        elif "AUTOINCREMENT" not in sql.upper() or (
            name == "kb_documents" and "'pending'" not in sql
        ):
            _rebuild(conn, name, ddl)


def _rebuild(conn: sqlite3.Connection, name: str, ddl: str) -> None:
    # SQLite can't add AUTOINCREMENT in place: copy into a new table and swap.
    # The caller turned foreign keys off (a no-op inside a transaction) and holds the
    # write lock, so dropping the old table doesn't cascade.
    temp = f"{name}_rebuild"
    old_cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({name})")}
    old_seq = 0
    if _table_sql(conn, "sqlite_sequence") is not None:
        seq_row = conn.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = ?", (name,)
        ).fetchone()
        old_seq = seq_row["seq"] if seq_row else 0
    conn.execute(f"DROP TABLE IF EXISTS {temp}")
    conn.execute(ddl.format(name=temp))
    new_cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({temp})")]
    cols = ", ".join(c for c in new_cols if c in old_cols)
    conn.execute(f"INSERT INTO {temp} ({cols}) SELECT {cols} FROM {name}")
    max_id = conn.execute(f"SELECT COALESCE(MAX(id), 0) FROM {temp}").fetchone()[0]
    seq = max(old_seq, max_id)
    cur = conn.execute("UPDATE sqlite_sequence SET seq = ? WHERE name = ?", (seq, temp))
    if cur.rowcount == 0:
        conn.execute("INSERT INTO sqlite_sequence (name, seq) VALUES (?, ?)", (temp, seq))
    conn.execute(f"DROP TABLE {name}")
    conn.execute(f"ALTER TABLE {temp} RENAME TO {name}")


INIT_BUSY_TIMEOUT_MS = 30000


def init_db(conn: sqlite3.Connection) -> None:
    """Create/upgrade the schema. Safe to run from several processes at once:
    everything happens inside one BEGIN IMMEDIATE transaction, with state re-checked
    after the write lock is held."""
    conn.commit()
    conn.execute(f"PRAGMA busy_timeout={INIT_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            for stmt in SCHEMA.split(";"):
                if stmt.strip():
                    conn.execute(stmt)
            _ensure_autoincrement(conn)
            for table, column, decl in MIGRATIONS:
                existing = {
                    r["name"] for r in conn.execute(f"PRAGMA table_info({table})")
                }
                if column not in existing:
                    try:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
                    except sqlite3.OperationalError as exc:
                        if "duplicate column" not in str(exc).lower():
                            raise
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    finally:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout=5000")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

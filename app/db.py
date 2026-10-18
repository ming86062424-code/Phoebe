import sqlite3
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
"""

# Knowledge base ids name files and Chroma collections, so they must never be reused
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
    status TEXT NOT NULL CHECK (status IN ('queued', 'parsing', 'indexing', 'ready', 'failed')),
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


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _table_sql(conn: sqlite3.Connection, name: str) -> str | None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row["sql"] if row else None


def _ensure_autoincrement(conn: sqlite3.Connection) -> None:
    """Create the AUTOINCREMENT tables, rebuilding ones made by older versions."""
    for name, ddl in AUTOINCREMENT_TABLES.items():
        sql = _table_sql(conn, name)
        if sql is None:
            conn.execute(ddl.format(name=name))
        elif "AUTOINCREMENT" not in sql.upper():
            _rebuild(conn, name, ddl)
    conn.commit()


def _rebuild(conn: sqlite3.Connection, name: str, ddl: str) -> None:
    # SQLite can't add AUTOINCREMENT in place: copy into a new table and swap.
    # Foreign keys are off so dropping the old table doesn't cascade.
    conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        temp = f"{name}_rebuild"
        old_cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({name})")}
        conn.execute(f"DROP TABLE IF EXISTS {temp}")
        conn.execute(ddl.format(name=temp))
        new_cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({temp})")]
        cols = ", ".join(c for c in new_cols if c in old_cols)
        conn.execute(f"INSERT INTO {temp} ({cols}) SELECT {cols} FROM {name}")
        conn.execute(f"DROP TABLE {name}")
        conn.execute(f"ALTER TABLE {temp} RENAME TO {name}")
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _ensure_autoincrement(conn)
    for table, column, decl in MIGRATIONS:
        existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    conn.commit()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

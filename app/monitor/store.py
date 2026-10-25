"""monitor.db: task events and resource samples for the monitoring page.

A separate SQLite file next to app.db: samples are written every few seconds and are
disposable, so they never compete with (or bloat) the accounts and conversations database.
Times are epoch seconds (float).
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from app.db import connect

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL NOT NULL,
    title TEXT NOT NULL,
    phases TEXT NOT NULL,
    details TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_started ON events (started_at);
CREATE TABLE IF NOT EXISTS resource_samples (
    ts REAL NOT NULL,
    source TEXT NOT NULL,
    kind TEXT NOT NULL,
    project TEXT NOT NULL,
    mem_mib REAL NOT NULL,
    mem_limit_mib REAL,
    cpu_pct REAL,
    gpu_util_pct REAL
);
CREATE INDEX IF NOT EXISTS resource_samples_ts ON resource_samples (ts);
CREATE TABLE IF NOT EXISTS collector_status (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_sample_at REAL,
    interval_seconds REAL NOT NULL,
    error TEXT NOT NULL
);
"""

EVENT_KINDS = ("document", "chat", "search")
# at most this many points per source in a chart; a bucket keeps its highest value
MAX_POINTS = 360


def monitor_db_path(app_db_path: str) -> str:
    """monitor.db sits next to app.db (in Docker: the phoebe-db volume)."""
    return str(Path(app_db_path).with_name("monitor.db"))


def open_db(app_db_path: str) -> sqlite3.Connection:
    conn = connect(monitor_db_path(app_db_path))
    conn.executescript(SCHEMA)
    return conn


@dataclass(frozen=True)
class Sample:
    source: str  # container name, or "GPU" / "GPU1" ...
    kind: str  # "container" or "gpu"
    mem_mib: float
    mem_limit_mib: float | None = None
    cpu_pct: float | None = None
    gpu_util_pct: float | None = None
    project: str = ""  # docker compose project ("phoebe"); empty for GPUs and loose containers


def record_event(
    app_db_path: str,
    kind: str,
    status: str,
    started_at: float,
    ended_at: float,
    title: str,
    phases: dict[str, float] | None = None,
    details: dict | None = None,
) -> None:
    """Store one finished task. Never raises: monitoring must not break the task itself."""
    try:
        conn = open_db(app_db_path)
        try:
            with conn:
                conn.execute(
                    "INSERT INTO events (kind, status, started_at, ended_at, title, phases, details)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        kind,
                        status,
                        started_at,
                        ended_at,
                        title,
                        json.dumps({k: round(v, 3) for k, v in (phases or {}).items()}),
                        json.dumps(details or {}, ensure_ascii=False),
                    ),
                )
        finally:
            conn.close()
    except Exception:
        logger.exception("無法記錄監控事件（%s）", kind)


def _event(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "kind": row["kind"],
        "status": row["status"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "duration": round(row["ended_at"] - row["started_at"], 3),
        "title": row["title"],
        "phases": json.loads(row["phases"]),
        "details": json.loads(row["details"]),
    }


def list_events(
    conn: sqlite3.Connection,
    kind: str | None = None,
    limit: int = 100,
    before_id: int | None = None,
    since: float | None = None,
    until: float | None = None,
) -> list[dict]:
    """Newest first. `since`/`until` keep events that overlap that time range."""
    where, args = [], []
    if kind:
        where.append("kind = ?")
        args.append(kind)
    if before_id is not None:
        where.append("id < ?")
        args.append(before_id)
    if since is not None:
        where.append("ended_at >= ?")
        args.append(since)
    if until is not None:
        where.append("started_at <= ?")
        args.append(until)
    sql = "SELECT * FROM events"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    return [_event(r) for r in conn.execute(sql, (*args, limit))]


def insert_samples(conn: sqlite3.Connection, ts: float, samples: list[Sample]) -> None:
    with conn:
        conn.executemany(
            "INSERT INTO resource_samples"
            " (ts, source, kind, project, mem_mib, mem_limit_mib, cpu_pct, gpu_util_pct)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (ts, s.source, s.kind, s.project, s.mem_mib, s.mem_limit_mib, s.cpu_pct,
                 s.gpu_util_pct)
                for s in samples
            ],
        )


def set_collector_status(
    conn: sqlite3.Connection, interval: float, last_sample_at: float | None, error: str
) -> None:
    with conn:
        conn.execute(
            "INSERT INTO collector_status (id, last_sample_at, interval_seconds, error)"
            " VALUES (1, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET"
            " last_sample_at = COALESCE(excluded.last_sample_at, last_sample_at),"
            " interval_seconds = excluded.interval_seconds, error = excluded.error",
            (last_sample_at, interval, error),
        )


def collector_status(conn: sqlite3.Connection) -> dict | None:
    row = conn.execute("SELECT * FROM collector_status WHERE id = 1").fetchone()
    if row is None:
        return None
    return {
        "last_sample_at": row["last_sample_at"],
        "interval_seconds": row["interval_seconds"],
        "error": row["error"],
    }


def _sample(row: sqlite3.Row) -> dict:
    return {
        "ts": row["ts"],
        "source": row["source"],
        "kind": row["kind"],
        "project": row["project"],
        "mem_mib": row["mem_mib"],
        "mem_limit_mib": row["mem_limit_mib"],
        "cpu_pct": row["cpu_pct"],
        "gpu_util_pct": row["gpu_util_pct"],
    }


def latest_samples(conn: sqlite3.Connection) -> list[dict]:
    """The most recent sampling round (every source in it)."""
    row = conn.execute("SELECT MAX(ts) AS ts FROM resource_samples").fetchone()
    if row["ts"] is None:
        return []
    rows = conn.execute(
        "SELECT * FROM resource_samples WHERE ts = ? ORDER BY kind, source", (row["ts"],)
    )
    return [_sample(r) for r in rows]


def list_samples(
    conn: sqlite3.Connection, since: float, source: str | None = None, limit: int = 500
) -> list[dict]:
    """Raw samples, newest first."""
    sql, args = "SELECT * FROM resource_samples WHERE ts >= ?", [since]
    if source:
        sql += " AND source = ?"
        args.append(source)
    sql += " ORDER BY ts DESC, source LIMIT ?"
    return [_sample(r) for r in conn.execute(sql, (*args, limit))]


def series(conn: sqlite3.Connection, since: float, until: float) -> dict:
    """Chart data: per source, at most MAX_POINTS buckets, each with the highest values
    of its samples (peaks are what the chart is for). Returns {"step", "sources": [...]}."""
    step = max((until - since) / MAX_POINTS, 1.0)
    rows = conn.execute(
        "SELECT source, kind, project, CAST((ts - ?) / ? AS INTEGER) AS bucket,"
        " MAX(ts) AS ts, MAX(mem_mib) AS mem_mib, MAX(mem_limit_mib) AS mem_limit_mib,"
        " MAX(cpu_pct) AS cpu_pct, MAX(gpu_util_pct) AS gpu_util_pct"
        " FROM resource_samples WHERE ts >= ? AND ts <= ?"
        " GROUP BY source, bucket ORDER BY source, bucket",
        (since, step, since, until),
    )
    sources: dict[str, dict] = {}
    for r in rows:
        entry = sources.setdefault(
            r["source"],
            {"source": r["source"], "kind": r["kind"], "project": r["project"], "points": []},
        )
        entry["points"].append(
            {
                "ts": r["ts"],
                "mem_mib": r["mem_mib"],
                "mem_limit_mib": r["mem_limit_mib"],
                "cpu_pct": r["cpu_pct"],
                "gpu_util_pct": r["gpu_util_pct"],
            }
        )
    return {"step": step, "sources": list(sources.values())}


def prune(
    conn: sqlite3.Connection, sample_hours: float, event_days: float, now: float | None = None
) -> None:
    now = time.time() if now is None else now
    with conn:
        conn.execute("DELETE FROM resource_samples WHERE ts < ?", (now - sample_hours * 3600,))
        conn.execute("DELETE FROM events WHERE ended_at < ?", (now - event_days * 86400,))

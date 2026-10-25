"""Monitoring page API (admins only): task events and resource samples from monitor.db."""

import sqlite3
import time
from collections.abc import Iterator

from fastapi import APIRouter, Depends, Query, Request

from app.auth import require_admin
from app.monitor import store

router = APIRouter(prefix="/api/monitor", dependencies=[Depends(require_admin)])

# the longest window the charts and the sample table offer (the samples' default retention)
MAX_MINUTES = 48 * 60


def get_monitor_db(request: Request) -> Iterator[sqlite3.Connection]:
    conn = store.open_db(request.app.state.settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


@router.get("/events")
def events(
    kind: str | None = Query(default=None, pattern="^(document|chat|search)$"),
    limit: int = Query(default=100, ge=1, le=500),
    before: int | None = Query(default=None, ge=1),
    conn: sqlite3.Connection = Depends(get_monitor_db),
):
    return {"events": store.list_events(conn, kind, limit, before)}


@router.get("/status")
def status(request: Request, conn: sqlite3.Connection = Depends(get_monitor_db)):
    """Whether the monitor service is sampling, and the latest value of every source."""
    collector = store.collector_status(conn)
    interval = request.app.state.settings.monitor_interval_seconds
    running = bool(
        collector
        and collector["last_sample_at"]
        # a few missed rounds are tolerated (a slow Docker API reply, a restart)
        and time.time() - collector["last_sample_at"] < max(collector["interval_seconds"], interval) * 4
    )
    return {"collector": collector, "running": running, "latest": store.latest_samples(conn)}


@router.get("/series")
def series(
    minutes: int = Query(default=60, ge=1, le=MAX_MINUTES),
    conn: sqlite3.Connection = Depends(get_monitor_db),
):
    """Chart data for the last `minutes`, with the events that overlap the window."""
    until = time.time()
    since = until - minutes * 60
    return {
        "since": since,
        "until": until,
        **store.series(conn, since, until),
        "events": store.list_events(conn, limit=500, since=since, until=until),
    }


@router.get("/samples")
def samples(
    minutes: int = Query(default=15, ge=1, le=MAX_MINUTES),
    source: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=500, ge=1, le=2000),
    conn: sqlite3.Connection = Depends(get_monitor_db),
):
    return {"samples": store.list_samples(conn, time.time() - minutes * 60, source, limit)}

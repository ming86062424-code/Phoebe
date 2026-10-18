import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi import HTTPException

from app.chat_stream import BUSY_DETAIL, claim_reply
from app.db import connect, init_db, now_iso
from app.deps import get_chat_client
from app.ollama_client import ChatDelta

pytestmark = pytest.mark.anyio

CREDS = {"username": "alice", "password": "password123"}


class GatedChatClient:
    """Streams one delta, then waits until the test opens the gate."""

    def __init__(self):
        self.gate = asyncio.Event()
        self.started = asyncio.Event()

    async def stream_chat(self, messages):
        yield ChatDelta("content", "第一段")
        self.started.set()
        await self.gate.wait()
        yield ChatDelta("content", "結束")


@asynccontextmanager
async def _client(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        await c.post("/api/auth/register", json=CREDS)
        await c.post("/api/auth/login", json=CREDS)
        yield c


async def _wait(event):
    await asyncio.wait_for(event.wait(), timeout=5)


def _soon(coro):
    return asyncio.wait_for(coro, timeout=5)


async def test_second_reply_in_same_conversation_is_409(app, settings):
    fake = GatedChatClient()
    app.dependency_overrides[get_chat_client] = lambda: fake
    async with _client(app) as c:
        cid = (await c.post("/api/conversations")).json()["id"]
        first = asyncio.create_task(
            c.post(f"/api/conversations/{cid}/messages", json={"content": "嗨"})
        )
        await _wait(fake.started)
        again = await _soon(c.post(f"/api/conversations/{cid}/regenerate"))
        assert again.status_code == 409
        other = await _soon(
            c.post(f"/api/conversations/{cid}/messages", json={"content": "再一次"})
        )
        assert other.status_code == 409
        fake.gate.set()
        done = await _soon(first)
        assert json.loads(done.text.splitlines()[-1])["type"] == "done"

        roles = [
            r["role"]
            for r in connect(settings.db_path).execute(
                "SELECT role FROM messages WHERE conversation_id = ? ORDER BY id", (cid,)
            )
        ]
        assert roles == ["user", "assistant"]
        # the lock is released once the reply finishes
        fake2 = GatedChatClient()
        fake2.gate.set()
        app.dependency_overrides[get_chat_client] = lambda: fake2
        r = await _soon(c.post(f"/api/conversations/{cid}/messages", json={"content": "下一題"}))
        assert r.status_code == 200


async def test_other_conversations_are_not_blocked(app):
    fake = GatedChatClient()
    app.dependency_overrides[get_chat_client] = lambda: fake
    async with _client(app) as c:
        a = (await c.post("/api/conversations")).json()["id"]
        b = (await c.post("/api/conversations")).json()["id"]
        first = asyncio.create_task(
            c.post(f"/api/conversations/{a}/messages", json={"content": "嗨"})
        )
        await _wait(fake.started)
        fake.gate.set()
        r = await _soon(c.post(f"/api/conversations/{b}/messages", json={"content": "嗨"}))
        assert r.status_code == 200
        await _soon(first)


@pytest.fixture
def two_conversations(settings):
    conn = connect(settings.db_path)
    init_db(conn)
    now = now_iso()
    conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) VALUES (1, 'u', 'x', ?)",
        (now,),
    )
    for cid in (1, 2):
        conn.execute(
            "INSERT INTO conversations (id, user_id, title, created_at, updated_at)"
            " VALUES (?, 1, '', ?, ?)",
            (cid, now, now),
        )
    conn.commit()
    conn.close()
    return settings.db_path


def test_second_claim_on_same_conversation_is_409(two_conversations):
    claim_reply(two_conversations, 1, 600)
    with pytest.raises(HTTPException) as e:
        claim_reply(two_conversations, 1, 600)
    assert e.value.status_code == 409
    assert e.value.detail == BUSY_DETAIL


def test_release_allows_claiming_again(two_conversations):
    release = claim_reply(two_conversations, 1, 600)
    release()
    claim_reply(two_conversations, 1, 600)


def test_release_is_idempotent(two_conversations):
    release = claim_reply(two_conversations, 1, 600)
    release()
    other = claim_reply(two_conversations, 1, 600)
    release()  # stale second call must not drop the newer claim
    with pytest.raises(HTTPException):
        claim_reply(two_conversations, 1, 600)
    other()


def test_expired_reply_lock_is_replaced(two_conversations):
    old = (datetime.now(timezone.utc) - timedelta(seconds=601)).isoformat()
    conn = connect(two_conversations)
    conn.execute("INSERT INTO active_replies (conversation_id, started_at) VALUES (1, ?)", (old,))
    conn.commit()
    conn.close()
    claim_reply(two_conversations, 1, 600)


def test_claims_on_different_conversations_do_not_conflict(two_conversations):
    claim_reply(two_conversations, 1, 600)
    claim_reply(two_conversations, 2, 600)


def test_heartbeat_keeps_reply_lock_alive_past_ttl(two_conversations):
    lock = claim_reply(two_conversations, 1, 1)
    for _ in range(4):
        time.sleep(0.5)
        assert lock.touch() is True
    with pytest.raises(HTTPException):  # 2 s old claim, but refreshed
        claim_reply(two_conversations, 1, 1)
    lock()  # release still matches the refreshed row
    claim_reply(two_conversations, 1, 1)


def test_reply_lock_without_heartbeat_expires(two_conversations):
    claim_reply(two_conversations, 1, 1)
    time.sleep(1.2)
    claim_reply(two_conversations, 1, 1)


def test_touch_does_not_resurrect_a_released_or_replaced_lock(two_conversations):
    lock = claim_reply(two_conversations, 1, 1)
    lock()
    assert lock.touch() is False
    time.sleep(1.1)
    first = claim_reply(two_conversations, 1, 1)
    time.sleep(1.1)
    second = claim_reply(two_conversations, 1, 1)  # replaced the expired first
    assert first.touch() is False
    second()


async def test_streaming_reply_refreshes_its_lock(app, settings, monkeypatch):
    from app import chat_stream

    monkeypatch.setattr(chat_stream, "REPLY_HEARTBEAT_SECONDS", 0.05)
    fake = GatedChatClient()
    app.dependency_overrides[get_chat_client] = lambda: fake
    async with _client(app) as c:
        cid = (await c.post("/api/conversations")).json()["id"]
        first = asyncio.create_task(
            c.post(f"/api/conversations/{cid}/messages", json={"content": "嗨"})
        )
        await _wait(fake.started)

        def started_at():
            conn = connect(settings.db_path)
            try:
                row = conn.execute(
                    "SELECT started_at FROM active_replies WHERE conversation_id = ?", (cid,)
                ).fetchone()
                return row["started_at"] if row else None
            finally:
                conn.close()

        before = started_at()
        await asyncio.sleep(0.5)
        assert started_at() > before
        fake.gate.set()
        await _soon(first)
        assert started_at() is None


class SlowStartChatClient:
    """First delta arrives only after `delay` seconds (cold-loading model)."""

    def __init__(self, delay):
        self.delay = delay

    async def stream_chat(self, messages):
        await asyncio.sleep(self.delay)
        yield ChatDelta("content", "好")


class UnavailableChatClient:
    async def stream_chat(self, messages):
        from app.ollama_client import OllamaUnavailable

        raise OllamaUnavailable("down")
        yield  # pragma: no cover


def _active_rows(db_path):
    conn = connect(db_path)
    try:
        return conn.execute("SELECT 1 FROM active_replies").fetchall()
    finally:
        conn.close()


async def test_lock_held_while_waiting_for_first_delta(settings, kb_services, monkeypatch):
    import dataclasses

    from app import chat_stream
    from app.main import create_app

    monkeypatch.setattr(chat_stream, "REPLY_HEARTBEAT_SECONDS", 0.2)
    app = create_app(dataclasses.replace(settings, reply_lock_ttl_seconds=1), kb_services)
    app.dependency_overrides[get_chat_client] = lambda: SlowStartChatClient(2.0)
    async with _client(app) as c:
        cid = (await c.post("/api/conversations")).json()["id"]
        first = asyncio.create_task(
            c.post(f"/api/conversations/{cid}/messages", json={"content": "嗨"})
        )
        await asyncio.sleep(1.6)  # past the TTL, still before the first delta
        again = await _soon(
            c.post(f"/api/conversations/{cid}/messages", json={"content": "再一次"})
        )
        assert again.status_code == 409
        done = await _soon(first)
        assert json.loads(done.text.splitlines()[-1])["type"] == "done"
        assert _active_rows(settings.db_path) == []


async def test_prestream_failure_releases_lock_and_stops_keep_alive(app, settings):
    app.dependency_overrides[get_chat_client] = lambda: UnavailableChatClient()
    async with _client(app) as c:
        cid = (await c.post("/api/conversations")).json()["id"]
        r = await _soon(c.post(f"/api/conversations/{cid}/messages", json={"content": "嗨"}))
        assert r.status_code == 503
        await asyncio.sleep(0.1)
        assert _active_rows(settings.db_path) == []
        assert not [t for t in asyncio.all_tasks() if "_keep_alive" in repr(t)]

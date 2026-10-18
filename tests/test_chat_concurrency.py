import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest

from app.db import connect
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

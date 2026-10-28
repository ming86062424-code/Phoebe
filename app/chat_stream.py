import asyncio
import json
import logging
import sqlite3
import threading
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from app import conversations
from app.config import REPLY_HEARTBEAT_SECONDS
from app.db import connect, now_iso
from app.llm import ChatClient, ChatDelta, ImagesNotSupported, LlmError, LlmUnavailable
from app.monitor.chat_trace import ChatTrace

logger = logging.getLogger(__name__)

UNAVAILABLE_DETAIL = "無法連線到 Ollama"
INTERRUPTED_DETAIL = "回覆中斷"
BUSY_DETAIL = "這個對話正在產生回覆，請稍候"
SAVE_FAILED_DETAIL = "回覆無法儲存"
# added to the system prompt when the model is asked again without the images
IMAGES_NOT_SENT_NOTE = "（圖片未提供給模型）"


class ReplyLock:
    """A held reply lock: call it to release, touch() to keep it from expiring."""

    def __init__(self, db_path: str, conversation_id: int, started_at: str):
        self.db_path = db_path
        self.conversation_id = conversation_id
        self.started_at = started_at
        self._mutex = threading.Lock()
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start_keep_alive(self, interval: float | None = None) -> None:
        """Refresh the lock in the background until it is released (call from the event loop)."""
        interval = REPLY_HEARTBEAT_SECONDS if interval is None else interval
        self._loop = asyncio.get_running_loop()
        self._task = self._loop.create_task(_keep_alive(self, interval))

    def _stop_keep_alive(self) -> None:
        task, loop = self._task, self._loop
        if task is None or loop is None:
            return
        try:  # release() may run in a background thread (background task)
            loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:  # loop already closed
            pass

    def touch(self) -> bool:
        """Refresh started_at; False when this lock no longer owns the row."""
        with self._mutex:
            new = now_iso()
            conn = connect(self.db_path)
            try:
                cur = conn.execute(
                    "UPDATE active_replies SET started_at = ? WHERE conversation_id = ? AND started_at = ?",
                    (new, self.conversation_id, self.started_at),
                )
                conn.commit()
            finally:
                conn.close()
            if cur.rowcount == 0:
                return False
            self.started_at = new
            return True

    def __call__(self) -> None:
        self._stop_keep_alive()
        # Match started_at so a stale release can't drop a newer claim.
        with self._mutex:
            conn = connect(self.db_path)
            try:
                conn.execute(
                    "DELETE FROM active_replies WHERE conversation_id = ? AND started_at = ?",
                    (self.conversation_id, self.started_at),
                )
                conn.commit()
            finally:
                conn.close()


def claim_reply(db_path: str, conversation_id: int, ttl_seconds: int) -> ReplyLock:
    """Mark the conversation as replying; returns the lock (callable: idempotent release).

    The lock lives in SQLite so it holds across Gunicorn processes. A lock not refreshed
    (see ReplyLock.touch) for ttl_seconds is treated as left over from a crashed Gunicorn process
    and replaced. Raises 409 when another reply for the same conversation is still running.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)).isoformat()
    started_at = now_iso()
    conn = connect(db_path)
    try:
        conn.execute(
            "DELETE FROM active_replies WHERE conversation_id = ? AND started_at < ?",
            (conversation_id, cutoff),
        )
        try:
            conn.execute(
                "INSERT INTO active_replies (conversation_id, started_at) VALUES (?, ?)",
                (conversation_id, started_at),
            )
        except sqlite3.IntegrityError as e:
            conn.rollback()
            raise HTTPException(status_code=409, detail=BUSY_DETAIL) from e
        conn.commit()
    finally:
        conn.close()
    return ReplyLock(db_path, conversation_id, started_at)


async def _keep_alive(lock: ReplyLock, interval: float) -> None:
    """Refresh the reply lock while the stream runs, so a dead process's lock expires fast."""
    touch = lock.touch
    while True:
        await asyncio.sleep(interval)
        try:
            if not await run_in_threadpool(touch):
                return
        except Exception:
            logger.exception("更新回覆鎖失敗")


def _with_images(history: list[dict], images: list[bytes]) -> list[dict]:
    """The history for the model, with images on the last user message (a copy)."""
    if not images or not history or history[-1]["role"] != "user":
        return history
    return [*history[:-1], {**history[-1], "images": images}]


async def _open_stream(
    client: ChatClient,
    history: list[dict],
    system: str | None,
    temperature: float | None,
    images: list[bytes],
) -> tuple[AsyncIterator[ChatDelta], ChatDelta | None]:
    """Start the reply and wait for its first delta.

    A model that rejects the images is asked once more without them. That can only
    happen here, before anything has reached the user; a later failure is reported as is.
    """
    stream = client.stream_chat(
        _with_images(history, images), system=system, temperature=temperature
    )
    try:
        return stream, await anext(stream)
    except StopAsyncIteration:
        return stream, None
    except ImagesNotSupported as e:
        if not images:
            raise
        logger.info("模型不接受圖片，改為不帶圖重試：%s", e)
    system = "\n\n".join(p for p in (system, IMAGES_NOT_SENT_NOTE) if p)
    stream = client.stream_chat(history, system=system, temperature=temperature)
    try:
        return stream, await anext(stream)
    except StopAsyncIteration:
        return stream, None


def _line(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


async def ndjson_reply(
    client: ChatClient,
    history: list[dict],
    conversation_id: int,
    db_path: str,
    release: Callable[[], None],
    sources: list[dict] | None = None,
    system: str | None = None,
    temperature: float | None = None,
    images: list[bytes] | None = None,
    trace: ChatTrace | None = None,
) -> StreamingResponse:
    """Stream the reply; release() runs once the reply ends, however it ends.

    When sources is given, they are emitted first and saved with the reply.
    images go to the model with the last user message only (never into the database).
    trace, when given, is finished with the reply's outcome (monitoring page).
    """
    try:
        return await _ndjson_reply(
            client,
            history,
            conversation_id,
            db_path,
            release,
            sources,
            system,
            temperature,
            images or [],
            trace,
        )
    except BaseException as e:
        release()
        if trace is not None:
            trace.finish("error", e.detail if isinstance(e, HTTPException) else str(e))
        raise


async def _ndjson_reply(
    client: ChatClient,
    history: list[dict],
    conversation_id: int,
    db_path: str,
    release: Callable[[], None],
    sources: list[dict] | None,
    system: str | None,
    temperature: float | None,
    images: list[bytes],
    trace: ChatTrace | None,
) -> StreamingResponse:
    if trace is not None:
        trace.llm_started()
    try:
        stream, first = await _open_stream(client, history, system, temperature, images)
    except LlmUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except LlmError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=UNAVAILABLE_DETAIL) from e

    async def body_iter():
        parts: list[str] = []

        def handle(delta: ChatDelta) -> str:
            if delta.kind == "content":
                parts.append(delta.text)
            if trace is not None:
                trace.delta(delta.kind)
            return _line({"type": delta.kind, "text": delta.text})

        def finish(status: str, error: str | None = None) -> None:
            if trace is not None:
                trace.finish(status, error)

        try:
            if sources is not None:
                yield _line({"type": "sources", "sources": sources})
            try:
                if first is not None:
                    yield handle(first)
                async for delta in stream:
                    yield handle(delta)
            except (LlmError, httpx.HTTPError) as e:
                finish("error", str(e) or INTERRUPTED_DETAIL)
                yield _line({"type": "error", "detail": str(e) or INTERRUPTED_DETAIL})
                return
            # The generator uses its own connection so it doesn't depend on the
            # request-scoped dependency's lifetime.
            save_conn = connect(db_path)
            try:
                message_id = conversations.add_message(
                    save_conn, conversation_id, "assistant", "".join(parts), sources
                )
            except sqlite3.Error:
                # e.g. the conversation was deleted in another tab mid-stream
                finish("error", SAVE_FAILED_DETAIL)
                yield _line({"type": "error", "detail": SAVE_FAILED_DETAIL})
                return
            finally:
                save_conn.close()
            finish("ok")
            yield _line({"type": "done", "message_id": message_id})
        finally:
            # the client went away mid-reply (finish() is a no-op after an outcome)
            finish("cancelled")
            release()

    def cleanup() -> None:
        if trace is not None:
            trace.finish("cancelled")
        release()

    # The background task covers a client that disconnects before the body starts.
    return StreamingResponse(
        body_iter(), media_type="application/x-ndjson", background=BackgroundTask(cleanup)
    )

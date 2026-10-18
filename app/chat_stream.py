import json
from collections.abc import Callable

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from app import conversations
from app.db import connect
from app.ollama_client import ChatDelta, OllamaClient, OllamaError, OllamaUnavailable

UNAVAILABLE_DETAIL = "無法連線到 Ollama"
INTERRUPTED_DETAIL = "回覆中斷"
BUSY_DETAIL = "這個對話正在產生回覆，請稍候"


def claim_reply(replying: set[int], conversation_id: int) -> Callable[[], None]:
    """Mark the conversation as replying; returns an idempotent release function.

    Raises 409 when another reply for the same conversation is still running.
    """
    if conversation_id in replying:
        raise HTTPException(status_code=409, detail=BUSY_DETAIL)
    replying.add(conversation_id)
    return lambda: replying.discard(conversation_id)


def _line(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


async def ndjson_reply(
    client: OllamaClient,
    history: list[dict],
    conversation_id: int,
    db_path: str,
    release: Callable[[], None],
) -> StreamingResponse:
    """Stream the reply; release() runs once the reply ends, however it ends."""
    try:
        return await _ndjson_reply(client, history, conversation_id, db_path, release)
    except BaseException:
        release()
        raise


async def _ndjson_reply(
    client: OllamaClient,
    history: list[dict],
    conversation_id: int,
    db_path: str,
    release: Callable[[], None],
) -> StreamingResponse:
    stream = client.stream_chat(history)
    try:
        first = await anext(stream)
    except StopAsyncIteration:
        first = None
    except OllamaUnavailable as e:
        raise HTTPException(status_code=503, detail=UNAVAILABLE_DETAIL) from e
    except OllamaError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=UNAVAILABLE_DETAIL) from e

    async def body_iter():
        parts: list[str] = []

        def handle(delta: ChatDelta) -> str:
            if delta.kind == "content":
                parts.append(delta.text)
            return _line({"type": delta.kind, "text": delta.text})

        try:
            try:
                if first is not None:
                    yield handle(first)
                async for delta in stream:
                    yield handle(delta)
            except (OllamaError, httpx.HTTPError) as e:
                yield _line({"type": "error", "detail": str(e) or INTERRUPTED_DETAIL})
                return
            # The generator uses its own connection so it doesn't depend on the
            # request-scoped dependency's lifetime.
            save_conn = connect(db_path)
            try:
                message_id = conversations.add_message(
                    save_conn, conversation_id, "assistant", "".join(parts)
                )
            finally:
                save_conn.close()
            yield _line({"type": "done", "message_id": message_id})
        finally:
            release()

    # The background task covers a client that disconnects before the body starts.
    return StreamingResponse(
        body_iter(), media_type="application/x-ndjson", background=BackgroundTask(release)
    )

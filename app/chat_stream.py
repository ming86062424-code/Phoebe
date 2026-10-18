import json

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app import conversations
from app.db import connect
from app.ollama_client import ChatDelta, OllamaClient, OllamaError, OllamaUnavailable

UNAVAILABLE_DETAIL = "無法連線到 Ollama"
INTERRUPTED_DETAIL = "回覆中斷"


def _line(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


async def ndjson_reply(
    client: OllamaClient, history: list[dict], conversation_id: int, db_path: str
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

    return StreamingResponse(body_iter(), media_type="application/x-ndjson")

import sqlite3

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from app import conversations
from app.auth import current_user
from app.config import Settings
from app.db import connect
from app.deps import get_chat_client, get_db, get_settings
from app.ollama_client import OllamaClient, OllamaError, OllamaUnavailable

router = APIRouter(prefix="/api")

MAX_CONTENT_LENGTH = 8000
UNAVAILABLE_DETAIL = "無法連線到 Ollama"


class MessageIn(BaseModel):
    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("content must not be blank")
        if len(stripped) > MAX_CONTENT_LENGTH:
            raise ValueError(f"content must be at most {MAX_CONTENT_LENGTH} characters")
        return v


@router.post("/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: int,
    body: MessageIn,
    user: dict = Depends(current_user),
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
    client: OllamaClient = Depends(get_chat_client),
):
    if conversations.get_conversation(conn, user["id"], conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conversations.add_message(conn, conversation_id, "user", body.content)
    conversations.set_title_if_empty(conn, conversation_id, body.content.strip()[:30])
    history = conversations.get_history(conn, conversation_id)

    stream = client.stream_chat(history)
    try:
        first = await anext(stream)
    except StopAsyncIteration:
        first = ""
    except OllamaUnavailable as e:
        raise HTTPException(status_code=503, detail=UNAVAILABLE_DETAIL) from e
    except OllamaError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=UNAVAILABLE_DETAIL) from e

    async def body_iter():
        parts = [first]
        if first:
            yield first
        async for chunk in stream:
            parts.append(chunk)
            yield chunk
        # The generator uses its own connection so it doesn't depend on the
        # request-scoped dependency's lifetime.
        save_conn = connect(settings.db_path)
        try:
            conversations.add_message(save_conn, conversation_id, "assistant", "".join(parts))
        finally:
            save_conn.close()

    return StreamingResponse(body_iter(), media_type="text/plain; charset=utf-8")

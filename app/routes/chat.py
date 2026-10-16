import sqlite3

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator

from app import conversations
from app.auth import current_user
from app.chat_stream import ndjson_reply
from app.config import Settings
from app.deps import get_chat_client, get_db, get_settings
from app.ollama_client import OllamaClient

router = APIRouter(prefix="/api")

MAX_CONTENT_LENGTH = 8000


class MessageIn(BaseModel):
    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("訊息不能是空白")
        if len(stripped) > MAX_CONTENT_LENGTH:
            raise ValueError(f"訊息最多 {MAX_CONTENT_LENGTH} 字")
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

    return await ndjson_reply(client, history, conversation_id, settings.db_path)


@router.post("/conversations/{conversation_id}/regenerate")
async def regenerate(
    conversation_id: int,
    user: dict = Depends(current_user),
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
    client: OllamaClient = Depends(get_chat_client),
):
    if conversations.get_conversation(conn, user["id"], conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    history = conversations.get_history(conn, conversation_id)
    if not history or history[-1]["role"] != "user":
        raise HTTPException(status_code=409, detail="沒有需要重新產生的回答")
    return await ndjson_reply(client, history, conversation_id, settings.db_path)

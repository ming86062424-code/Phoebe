import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from app import conversations
from app.auth import current_user
from app.deps import get_db
from app.kb import store as kb_store

router = APIRouter(prefix="/api")


@router.get("/conversations")
def list_all(user: dict = Depends(current_user), conn: sqlite3.Connection = Depends(get_db)):
    return conversations.list_conversations(conn, user["id"])


@router.post("/conversations", status_code=201)
def create(user: dict = Depends(current_user), conn: sqlite3.Connection = Depends(get_db)):
    return conversations.create_conversation(conn, user["id"])


@router.get("/conversations/{conversation_id}")
def get_one(
    conversation_id: int,
    user: dict = Depends(current_user),
    conn: sqlite3.Connection = Depends(get_db),
):
    conv = conversations.get_conversation(conn, user["id"], conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete(
    conversation_id: int,
    user: dict = Depends(current_user),
    conn: sqlite3.Connection = Depends(get_db),
):
    if not conversations.delete_conversation(conn, user["id"], conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return Response(status_code=204)


class ConversationPatch(BaseModel):
    kb_id: int | None


@router.patch("/conversations/{conversation_id}")
def update(
    conversation_id: int,
    body: ConversationPatch,
    user: dict = Depends(current_user),
    conn: sqlite3.Connection = Depends(get_db),
):
    if conversations.get_conversation(conn, user["id"], conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if body.kb_id is not None and kb_store.get_kb(conn, user["id"], body.kb_id) is None:
        raise HTTPException(status_code=404, detail="找不到知識庫")
    conversations.set_kb(conn, conversation_id, body.kb_id)
    return conversations.get_conversation(conn, user["id"], conversation_id)

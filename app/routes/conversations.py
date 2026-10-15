import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Response

from app import conversations
from app.auth import current_user
from app.deps import get_db

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

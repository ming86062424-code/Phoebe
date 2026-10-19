import sqlite3
from collections.abc import Iterator

from fastapi import Depends, HTTPException, Request

from app.config import Settings
from app.db import connect
from app.kb.services import KbServices
from app.llm import ChatClient
from app.llm.factory import build_chat_client
from app.llm_settings import load_llm_settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> Iterator[sqlite3.Connection]:
    conn = connect(request.app.state.settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


def get_kb_services(request: Request) -> KbServices:
    return request.app.state.kb


def get_chat_client(
    settings: Settings = Depends(get_settings),
    conn: sqlite3.Connection = Depends(get_db),
) -> ChatClient:
    # Read per request (no caching) so another worker's settings change applies at once.
    llm = load_llm_settings(conn, settings)
    # Only the key check is cheap; Ollama problems surface as 503 from the stream itself.
    if llm.llm_provider == "gemini" and not settings.gemini_api_key:
        raise HTTPException(status_code=503, detail="Gemini 目前不可用：未設定 GEMINI_API_KEY")
    return build_chat_client(settings, llm)

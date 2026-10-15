import sqlite3
from collections.abc import Iterator

from fastapi import Depends, Request

from app.config import Settings
from app.db import connect
from app.ollama_client import OllamaClient


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> Iterator[sqlite3.Connection]:
    conn = connect(request.app.state.settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


def get_chat_client(settings: Settings = Depends(get_settings)) -> OllamaClient:
    return OllamaClient(settings.ollama_base_url, settings.chat_model)

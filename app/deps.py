import sqlite3
from collections.abc import Iterator

from fastapi import Depends, Request

from app.config import Settings
from app.db import connect
from app.kb.services import KbServices
from app.llm import ChatClient, OllamaChatClient


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


def get_chat_client(settings: Settings = Depends(get_settings)) -> ChatClient:
    return OllamaChatClient(
        settings.ollama_base_url, settings.chat_model, think=settings.chat_think
    )

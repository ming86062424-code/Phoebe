import sqlite3
from collections.abc import AsyncIterator, Iterator

from fastapi import Depends, Request

from app.config import Settings
from app.db import connect
from app.kb.services import KbServices
from app.llm import ChatClient
from app.llm.base import ChatDelta, LlmUnavailable
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


GEMINI_KEY_MISSING = "Gemini 目前不可用：未設定 GEMINI_API_KEY"


class UnavailableChatClient:
    """Stands in when the chosen provider cannot work at all (no Gemini key).

    It fails when the reply starts, not as a dependency: the user's message is saved
    first, as with any other 503, so the chat page can always retry with regenerate.
    """

    def __init__(self, detail: str):
        self.detail = detail

    async def stream_chat(
        self, messages: list[dict], *, system: str | None = None, temperature: float | None = None
    ) -> AsyncIterator[ChatDelta]:
        raise LlmUnavailable(self.detail)
        yield  # makes this an async generator


def get_chat_client(
    settings: Settings = Depends(get_settings),
    conn: sqlite3.Connection = Depends(get_db),
) -> ChatClient:
    # Read per request (no caching) so another Gunicorn process's settings change applies at once.
    llm = load_llm_settings(conn, settings)
    # Only the key check is cheap; Ollama problems surface as 503 from the stream itself.
    if llm.llm_provider == "gemini" and not settings.gemini_api_key:
        return UnavailableChatClient(GEMINI_KEY_MISSING)
    return build_chat_client(settings, llm)

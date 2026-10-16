from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import Settings
from app.db import connect, init_db
from app.routes import auth, chat, conversations

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI()
    app.state.settings = settings
    conn = connect(settings.db_path)
    try:
        init_db(conn)
    finally:
        conn.close()
    app.include_router(auth.router)
    app.include_router(conversations.router)
    app.include_router(chat.router)
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app

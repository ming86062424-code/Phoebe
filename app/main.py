from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings
from app.db import connect, init_db
from app.routes import auth, chat, conversations

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
    dist = Path(settings.frontend_dist)
    if (dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    else:

        @app.get("/")
        def frontend_missing() -> PlainTextResponse:
            return PlainTextResponse("前端尚未建置，請在 frontend/ 執行 npm run build")

    return app

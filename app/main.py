from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings
from app.db import connect, init_db
from app.kb.services import KbServices, build_kb_services
from app.routes import auth, chat, conversations, kb

def create_app(
    settings: Settings | None = None, kb_services: KbServices | None = None
) -> FastAPI:
    settings = settings or Settings.from_env()
    conn = connect(settings.db_path)
    try:
        init_db(conn)
    finally:
        conn.close()
    kb_services = kb_services or build_kb_services(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        kb_services.worker.start()
        yield
        kb_services.worker.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.settings = settings
    app.state.kb = kb_services
    # ids of conversations with a reply in progress (see chat_stream.claim_reply)
    app.state.replying = set()
    app.include_router(auth.router)
    app.include_router(conversations.router)
    app.include_router(chat.router)
    app.include_router(kb.router)
    dist = Path(settings.frontend_dist)
    if (dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    else:

        @app.get("/")
        def frontend_missing() -> PlainTextResponse:
            return PlainTextResponse("前端尚未建置，請在 frontend/ 執行 npm run build")

    return app

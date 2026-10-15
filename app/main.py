from fastapi import FastAPI

from app.config import Settings
from app.db import connect, init_db
from app.routes import auth, conversations


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
    return app

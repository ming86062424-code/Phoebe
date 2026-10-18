import os

bind = "0.0.0.0:8000"
worker_class = "uvicorn_worker.UvicornWorker"
workers = int(os.environ.get("WEB_CONCURRENCY", "2"))
timeout = 120
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"


def on_starting(server):
    """Create/upgrade the schema once in the master, before workers fork."""
    from app.config import Settings
    from app.db import connect, init_db

    conn = connect(Settings.from_env().db_path)
    try:
        init_db(conn)
    finally:
        conn.close()

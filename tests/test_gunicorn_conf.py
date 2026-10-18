import runpy
from pathlib import Path

CONF = Path(__file__).resolve().parent.parent / "docker" / "gunicorn.conf.py"


def test_gunicorn_conf_values(monkeypatch):
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    conf = runpy.run_path(str(CONF))
    assert conf["bind"] == "0.0.0.0:8000"
    assert conf["worker_class"] == "uvicorn_worker.UvicornWorker"
    assert conf["workers"] == 2
    assert conf["timeout"] == 120
    assert conf["graceful_timeout"] == 30
    assert conf["keepalive"] == 5
    assert conf["accesslog"] == "-"

    monkeypatch.setenv("WEB_CONCURRENCY", "3")
    assert runpy.run_path(str(CONF))["workers"] == 3


def test_on_starting_creates_schema(monkeypatch, tmp_path):
    import sqlite3

    db = tmp_path / "sub" / "app.db"
    monkeypatch.setenv("DB_PATH", str(db))
    conf = runpy.run_path(str(CONF))
    conf["on_starting"](None)
    names = {
        r[0]
        for r in sqlite3.connect(db).execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"users", "knowledge_bases", "kb_documents"} <= names

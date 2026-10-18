import sqlite3
from pathlib import Path

import pytest

from app.config import Settings
from app.db import connect, init_db, now_iso


def _seed_message(conn):
    now = now_iso()
    conn.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES ('u', 'h', ?)", (now,)
    )
    conn.execute(
        "INSERT INTO conversations (user_id, title, created_at, updated_at) VALUES (1, 't', ?, ?)",
        (now, now),
    )


def _insert_message(conn, role):
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (1, ?, 'hi', ?)",
        (role, now_iso()),
    )


def test_init_db_creates_tables(tmp_path):
    conn = connect(str(tmp_path / "sub" / "t.db"))
    init_db(conn)
    names = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"users", "sessions", "conversations", "messages"} <= names


def test_init_db_is_idempotent(tmp_path):
    conn = connect(str(tmp_path / "t.db"))
    init_db(conn)
    init_db(conn)


def test_foreign_keys_enabled(tmp_path):
    conn = connect(str(tmp_path / "t.db"))
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_deleting_conversation_cascades_messages(tmp_path):
    conn = connect(str(tmp_path / "t.db"))
    init_db(conn)
    _seed_message(conn)
    _insert_message(conn, "user")
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    conn.execute("DELETE FROM conversations WHERE id = 1")
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


def test_message_role_check_rejects_invalid_role(tmp_path):
    conn = connect(str(tmp_path / "t.db"))
    init_db(conn)
    _seed_message(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_message(conn, "system")


def test_settings_from_env_defaults_and_override(monkeypatch):
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    monkeypatch.delenv("EVAL_JUDGE_MODEL", raising=False)
    assert Settings.from_env().chat_model == "gemma4:e4b"
    assert Settings.from_env().eval_judge_model == "qwen3.5:4b"
    monkeypatch.setenv("SESSION_DAYS", "3")
    assert Settings.from_env().session_days == 3


def test_default_db_path_is_absolute_project_data(monkeypatch):
    monkeypatch.delenv("DB_PATH", raising=False)
    path = Path(Settings.from_env().db_path)
    assert path.is_absolute()
    assert path.parts[-2:] == ("data", "app.db")


def test_settings_chat_think_env(monkeypatch):
    monkeypatch.setenv("CHAT_THINK", "FALSE")
    assert Settings.from_env().chat_think is False
    monkeypatch.setenv("CHAT_THINK", "0")
    assert Settings.from_env().chat_think is False
    monkeypatch.delenv("CHAT_THINK")
    assert Settings.from_env().chat_think is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off", "OFF", " No "])
def test_settings_chat_think_falsy_values(monkeypatch, value):
    monkeypatch.setenv("CHAT_THINK", value)
    assert Settings.from_env().chat_think is False


@pytest.mark.parametrize("value", ["true", "1", "yes", "on"])
def test_settings_chat_think_truthy_values(monkeypatch, value):
    monkeypatch.setenv("CHAT_THINK", value)
    assert Settings.from_env().chat_think is True

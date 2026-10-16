import secrets
import sqlite3
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import Depends, HTTPException, Request

from app.db import now_iso
from app.deps import get_db


class UsernameTaken(Exception):
    pass


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        return False


# Valid hash of a random string, used so unknown usernames cost the same as wrong passwords.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def create_user(conn: sqlite3.Connection, username: str, password: str) -> int:
    try:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, hash_password(password), now_iso()),
        )
    except sqlite3.IntegrityError:
        raise UsernameTaken(username) from None
    conn.commit()
    return cur.lastrowid


def authenticate(conn: sqlite3.Connection, username: str, password: str) -> int | None:
    row = conn.execute(
        "SELECT id, password_hash FROM users WHERE username = ?", (username,)
    ).fetchone()
    if row is None:
        verify_password(password, _DUMMY_HASH)
        return None
    return row["id"] if verify_password(password, row["password_hash"]) else None


def create_session(conn: sqlite3.Connection, user_id: int, days: int) -> str:
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    conn.execute(
        "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (token, user_id, now.isoformat(), (now + timedelta(days=days)).isoformat()),
    )
    conn.commit()
    return token


def delete_session(conn: sqlite3.Connection, token: str) -> None:
    conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()


def current_user(request: Request, conn: sqlite3.Connection = Depends(get_db)) -> dict:
    token = request.cookies.get("session")
    if token:
        row = conn.execute(
            "SELECT u.id, u.username, s.expires_at FROM sessions s "
            "JOIN users u ON u.id = s.user_id WHERE s.token = ?",
            (token,),
        ).fetchone()
        if row and datetime.fromisoformat(row["expires_at"]) > datetime.now(timezone.utc):
            return {"id": row["id"], "username": row["username"]}
    raise HTTPException(status_code=401, detail="Not authenticated")

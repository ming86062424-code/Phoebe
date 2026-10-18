import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, field_validator

from app import auth
from app.config import Settings
from app.deps import get_db, get_settings

router = APIRouter(prefix="/api")


class RegisterBody(BaseModel):
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def username_length(cls, v: str) -> str:
        if not 3 <= len(v) <= 32:
            raise ValueError("帳號長度需為 3 到 32 個字元")
        return v

    @field_validator("password")
    @classmethod
    def password_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("密碼至少需要 8 個字元")
        # bcrypt ignores everything past 72 bytes
        if len(v.encode()) > 72:
            raise ValueError("密碼最多 72 個位元組")
        return v


class LoginBody(BaseModel):
    username: str
    password: str


@router.post("/auth/register", status_code=201)
def register(body: RegisterBody, conn: sqlite3.Connection = Depends(get_db)):
    try:
        user_id = auth.create_user(conn, body.username, body.password)
    except auth.UsernameTaken:
        raise HTTPException(status_code=409, detail="這個帳號已經有人使用") from None
    return {"id": user_id, "username": body.username}


@router.post("/auth/login")
def login(
    body: LoginBody,
    response: Response,
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    user_id = auth.authenticate(conn, body.username, body.password)
    if user_id is None:
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
    token = auth.create_session(conn, user_id, settings.session_days)
    response.set_cookie(
        "session",
        token,
        max_age=settings.session_days * 86400,
        httponly=True,
        samesite="lax",
        secure=False,
    )
    return {"id": user_id, "username": body.username}


@router.post("/auth/logout", status_code=204)
def logout(request: Request, conn: sqlite3.Connection = Depends(get_db)):
    token = request.cookies.get("session")
    if token:
        auth.delete_session(conn, token)
    response = Response(status_code=204)
    response.delete_cookie("session")
    return response


@router.get("/me")
def me(user: dict = Depends(auth.current_user)):
    return user

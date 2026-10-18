import sqlite3
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict

from app import llm_settings
from app.auth import require_admin
from app.config import Settings
from app.deps import get_db, get_settings
from app.llm import LlmError, OllamaChatClient
from app.llm.factory import PROVIDER_LABELS, provider_availability

router = APIRouter(prefix="/api/settings", dependencies=[Depends(require_admin)])


class SettingsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_provider: str | None = None
    ollama_model: str | None = None
    gemini_model: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None


async def _response(
    conn: sqlite3.Connection, settings: Settings, availability: dict | None = None
) -> dict:
    current = await run_in_threadpool(llm_settings.load_llm_settings, conn, settings)
    availability = availability or await provider_availability(settings)
    return {
        **llm_settings.as_dict(current),
        "providers": {
            name: {"available": ok, "reason": reason}
            for name, (ok, reason) in availability.items()
        },
    }


@router.get("")
async def get_settings_route(
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    return await _response(conn, settings)


@router.put("")
async def put_settings(
    body: SettingsBody,
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    values = body.model_dump(exclude_unset=True)
    try:
        values = llm_settings.validate_values(values)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
    availability = await provider_availability(settings)
    provider = values.get("llm_provider")
    current = await run_in_threadpool(llm_settings.load_llm_settings, conn, settings)
    if provider is not None and provider != current.llm_provider:
        ok, reason = availability[provider]
        if not ok:
            raise HTTPException(
                status_code=409,
                detail=f"{PROVIDER_LABELS[provider]} 目前不可用：{reason}",
            )
    await run_in_threadpool(llm_settings.save_llm_settings, conn, settings, values)
    return await _response(conn, settings, availability)


@router.get("/models")
async def list_models(
    provider: Literal["ollama", "gemini"],
    settings: Settings = Depends(get_settings),
):
    if provider == "ollama":
        client = OllamaChatClient(
            settings.ollama_base_url, settings.chat_model, think=settings.chat_think
        )
        try:
            return {"models": await client.list_models()}
        except LlmError as e:
            raise HTTPException(status_code=503, detail=str(e)) from None
    if not settings.gemini_api_key:
        raise HTTPException(status_code=503, detail="未設定 GEMINI_API_KEY")
    # Task 3 replaces this branch with GeminiChatClient.list_models()
    raise HTTPException(status_code=503, detail="無法連線到 Gemini")

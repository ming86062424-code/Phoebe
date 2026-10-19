import json
import sqlite3
from dataclasses import asdict, dataclass

from app.config import Settings
from app.db import now_iso

PROVIDERS = ("ollama", "gemini")
SYSTEM_PROMPT_MAX = 4000
KEYS = ("llm_provider", "ollama_model", "gemini_model", "system_prompt", "temperature")


@dataclass(frozen=True)
class LlmSettings:
    llm_provider: str
    ollama_model: str
    gemini_model: str
    system_prompt: str
    temperature: float | None


def default_values(settings: Settings) -> dict:
    return {
        "llm_provider": settings.llm_provider,
        "ollama_model": settings.chat_model,
        "gemini_model": settings.gemini_model,
        "system_prompt": "",
        "temperature": None,
    }


def validate_values(values: dict) -> dict:
    """Return the normalized values, or raise ValueError with a Chinese message."""
    unknown = set(values) - set(KEYS)
    if unknown:
        raise ValueError(f"未知的設定項目：{', '.join(sorted(unknown))}")
    out = dict(values)
    if "llm_provider" in out and out["llm_provider"] not in PROVIDERS:
        raise ValueError("供應商必須是 ollama 或 gemini")
    for key in ("ollama_model", "gemini_model"):
        if key in out:
            value = out[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError("模型名稱不可為空")
            out[key] = value.strip()
    if "system_prompt" in out:
        value = out["system_prompt"]
        if not isinstance(value, str):
            raise ValueError("系統提示詞必須是文字")
        if len(value) > SYSTEM_PROMPT_MAX:
            raise ValueError(f"系統提示詞最多 {SYSTEM_PROMPT_MAX} 字")
    if "temperature" in out and out["temperature"] is not None:
        value = out["temperature"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("temperature 必須是數字或 null")
        if not 0 <= value <= 2:
            raise ValueError("temperature 必須介於 0 到 2")
        out["temperature"] = float(value)
    return out


def load_llm_settings(conn: sqlite3.Connection, settings: Settings) -> LlmSettings:
    values = default_values(settings)
    for row in conn.execute("SELECT key, value FROM app_settings"):
        if row["key"] in values:
            values[row["key"]] = json.loads(row["value"])
    return LlmSettings(**values)


def save_llm_settings(
    conn: sqlite3.Connection, settings: Settings, values: dict
) -> LlmSettings:
    clean = validate_values(values)
    now = now_iso()
    with conn:
        for key, value in clean.items():
            conn.execute(
                "INSERT INTO app_settings (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at",
                (key, json.dumps(value, ensure_ascii=False), now),
            )
    return load_llm_settings(conn, settings)


def as_dict(llm: LlmSettings) -> dict:
    return asdict(llm)

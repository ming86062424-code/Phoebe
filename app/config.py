import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FALSY_VALUES = ("false", "0", "no", "off")
# how often a streaming reply refreshes its lock; REPLY_LOCK_TTL_SECONDS must exceed it
REPLY_HEARTBEAT_SECONDS = 30


def env_file_path() -> Path:
    return Path(os.environ.get("ENV_FILE") or PROJECT_ROOT / ".env")


def _choice(name: str, default: str, allowed: tuple[str, str]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in allowed:
        raise ValueError(f"{name} 必須是 {allowed[0]} 或 {allowed[1]}")
    return value


def _non_negative_int(name: str, raw: str) -> int:
    try:
        value = int(raw.strip())
    except ValueError:
        value = -1
    if value < 0:
        raise ValueError(f"{name} 必須是大於或等於 0 的整數")
    return value


def _check_admin(username: str, password: str) -> None:
    """The admin account from .env follows the same rules as registration."""
    if not username and not password:
        return
    if not username or not password:
        raise ValueError("ADMIN_USERNAME 與 ADMIN_PASSWORD 必須同時設定")
    if not 3 <= len(username) <= 32:
        raise ValueError("ADMIN_USERNAME 長度需為 3 到 32 個字元")
    if len(password) < 8:
        raise ValueError("ADMIN_PASSWORD 至少需要 8 個字元")
    # bcrypt ignores everything past 72 bytes
    if len(password.encode()) > 72:
        raise ValueError("ADMIN_PASSWORD 最多 72 個位元組")


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "gemma4:e4b"
    eval_judge_model: str = "qwen3.5:4b"
    db_path: str = str(PROJECT_ROOT / "data" / "app.db")
    session_days: int = 7
    chat_think: bool = True
    frontend_dist: str = str(PROJECT_ROOT / "frontend" / "dist")
    embedder_url: str = "http://localhost:8002"
    kb_dir: str = str(PROJECT_ROOT / "data" / "kb")
    kb_top_k: int = 5
    kb_max_images: int = 4
    max_upload_mb: int = 50
    ocr_model: str = "lightonai/LightOnOCR-3-0.8B"
    liteparse_ocr_language: str = "chi_tra+eng"
    kb_worker_mode: str = "embedded"
    ocr_backend: str = "transformers"
    vllm_base_url: str = "http://localhost:8001/v1"
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = field(default="", repr=False)
    milvus_collection: str = "phoebe_chunks"
    reply_lock_ttl_seconds: int = 90
    kb_worker_poll_seconds: float = 2.0
    kb_stale_seconds: int = 120
    kb_heartbeat_seconds: float = 30.0
    admin_username: str = ""
    admin_password: str = field(default="", repr=False)
    gemini_api_key: str = field(default="", repr=False)
    gemini_model: str = "gemini-2.5-flash"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(env_file_path(), override=False)
        d = cls()
        settings = cls(
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", d.ollama_base_url),
            chat_model=os.environ.get("CHAT_MODEL", d.chat_model),
            eval_judge_model=os.environ.get("EVAL_JUDGE_MODEL", d.eval_judge_model),
            db_path=os.environ.get("DB_PATH", d.db_path),
            session_days=int(os.environ.get("SESSION_DAYS", d.session_days)),
            chat_think=os.environ.get("CHAT_THINK", "true").strip().lower()
            not in FALSY_VALUES,
            frontend_dist=os.environ.get("FRONTEND_DIST", d.frontend_dist),
            embedder_url=os.environ.get("EMBEDDER_URL", d.embedder_url).strip(),
            kb_dir=os.environ.get("KB_DIR", d.kb_dir),
            kb_top_k=int(os.environ.get("KB_TOP_K", d.kb_top_k)),
            kb_max_images=_non_negative_int(
                "KB_MAX_IMAGES", os.environ.get("KB_MAX_IMAGES", str(d.kb_max_images))
            ),
            max_upload_mb=int(os.environ.get("MAX_UPLOAD_MB", d.max_upload_mb)),
            ocr_model=os.environ.get("OCR_MODEL", d.ocr_model),
            liteparse_ocr_language=os.environ.get(
                "LITEPARSE_OCR_LANGUAGE", d.liteparse_ocr_language
            ),
            kb_worker_mode=_choice(
                "KB_WORKER_MODE", d.kb_worker_mode, ("embedded", "external")
            ),
            ocr_backend=_choice("OCR_BACKEND", d.ocr_backend, ("transformers", "vllm")),
            vllm_base_url=os.environ.get("VLLM_BASE_URL", d.vllm_base_url),
            milvus_uri=os.environ.get("MILVUS_URI", d.milvus_uri),
            milvus_token=os.environ.get("MILVUS_TOKEN", d.milvus_token).strip(),
            milvus_collection=os.environ.get("MILVUS_COLLECTION", d.milvus_collection),
            reply_lock_ttl_seconds=int(
                os.environ.get("REPLY_LOCK_TTL_SECONDS", d.reply_lock_ttl_seconds)
            ),
            kb_worker_poll_seconds=float(
                os.environ.get("KB_WORKER_POLL_SECONDS", d.kb_worker_poll_seconds)
            ),
            kb_stale_seconds=int(os.environ.get("KB_STALE_SECONDS", d.kb_stale_seconds)),
            kb_heartbeat_seconds=float(
                os.environ.get("KB_HEARTBEAT_SECONDS", d.kb_heartbeat_seconds)
            ),
            admin_username=os.environ.get("ADMIN_USERNAME", "").strip(),
            admin_password=os.environ.get("ADMIN_PASSWORD", ""),
            gemini_api_key=os.environ.get("GEMINI_API_KEY", d.gemini_api_key).strip(),
            gemini_model=os.environ.get("GEMINI_MODEL", d.gemini_model),
        )
        if settings.kb_heartbeat_seconds >= settings.kb_stale_seconds:
            raise ValueError("KB_HEARTBEAT_SECONDS 必須小於 KB_STALE_SECONDS")
        if os.environ.get("ADMIN_USERNAMES", "").strip():
            raise ValueError(
                "ADMIN_USERNAMES 已不再使用，請改設 ADMIN_USERNAME 與 ADMIN_PASSWORD"
            )
        _check_admin(settings.admin_username, settings.admin_password)
        if settings.reply_lock_ttl_seconds <= REPLY_HEARTBEAT_SECONDS:
            raise ValueError(
                f"REPLY_LOCK_TTL_SECONDS 必須大於 {REPLY_HEARTBEAT_SECONDS}（回覆鎖的更新間隔）"
            )
        return settings

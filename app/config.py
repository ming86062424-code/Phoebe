import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "gemma4:e4b"
    eval_judge_model: str = "qwen3.5:4b"
    db_path: str = str(PROJECT_ROOT / "data" / "app.db")
    session_days: int = 7
    chat_think: bool = True
    frontend_dist: str = str(PROJECT_ROOT / "frontend" / "dist")

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", d.ollama_base_url),
            chat_model=os.environ.get("CHAT_MODEL", d.chat_model),
            eval_judge_model=os.environ.get("EVAL_JUDGE_MODEL", d.eval_judge_model),
            db_path=os.environ.get("DB_PATH", d.db_path),
            session_days=int(os.environ.get("SESSION_DAYS", d.session_days)),
            chat_think=os.environ.get("CHAT_THINK", "true").strip().lower()
            not in ("false", "0"),
            frontend_dist=os.environ.get("FRONTEND_DIST", d.frontend_dist),
        )

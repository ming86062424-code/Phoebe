import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FALSY_VALUES = ("false", "0", "no", "off")


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "gemma4:e4b"
    eval_judge_model: str = "qwen3.5:4b"
    db_path: str = str(PROJECT_ROOT / "data" / "app.db")
    session_days: int = 7
    chat_think: bool = True
    frontend_dist: str = str(PROJECT_ROOT / "frontend" / "dist")
    embed_model: str = "embeddinggemma-2"
    kb_dir: str = str(PROJECT_ROOT / "data" / "kb")
    chroma_dir: str = str(PROJECT_ROOT / "data" / "chroma")
    kb_top_k: int = 5
    max_upload_mb: int = 50
    ocr_model: str = "lightonai/LightOnOCR-3-0.8B"
    liteparse_ocr_language: str = "chi_tra+eng"

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
            not in FALSY_VALUES,
            frontend_dist=os.environ.get("FRONTEND_DIST", d.frontend_dist),
            embed_model=os.environ.get("EMBED_MODEL", d.embed_model),
            kb_dir=os.environ.get("KB_DIR", d.kb_dir),
            chroma_dir=os.environ.get("CHROMA_DIR", d.chroma_dir),
            kb_top_k=int(os.environ.get("KB_TOP_K", d.kb_top_k)),
            max_upload_mb=int(os.environ.get("MAX_UPLOAD_MB", d.max_upload_mb)),
            ocr_model=os.environ.get("OCR_MODEL", d.ocr_model),
            liteparse_ocr_language=os.environ.get(
                "LITEPARSE_OCR_LANGUAGE", d.liteparse_ocr_language
            ),
        )

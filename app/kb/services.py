from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from llama_index.core.base.embeddings.base import BaseEmbedding

from app.config import Settings
from app.db import connect
from app.kb.index import KnowledgeIndex
from app.kb.parsers.base import Parser
from app.kb.worker import IngestWorker


WORKER_DOWN_REASON = "文件解析服務（worker）未啟動"
# the worker republishes every 15 s; older than this means it is gone
PARSER_STATUS_MAX_AGE_SECONDS = 45


@dataclass
class KbServices:
    index: KnowledgeIndex
    parsers: dict[str, Parser]
    kb_dir: Path
    db_path: str = ""
    external: bool = False
    worker: IngestWorker = field(init=False)

    def file_path(self, kb_id: int, doc_id: int) -> Path:
        return self.kb_dir / str(kb_id) / f"{doc_id}.pdf"

    def availability(self, parser_id: str) -> tuple[bool, str]:
        """Whether a parser can be used right now.

        With an external worker the parsers' dependencies (torch, vLLM) live in the
        worker's container, so the web process reads the status the worker publishes.
        """
        if not self.external:
            return self.parsers[parser_id].availability()
        down = (False, WORKER_DOWN_REASON)
        try:
            conn = connect(self.db_path)
            try:
                row = conn.execute(
                    "SELECT available, reason, updated_at FROM parser_status WHERE parser_id = ?",
                    (parser_id,),
                ).fetchone()
            finally:
                conn.close()
            if row is None:
                return down
            age = datetime.now(timezone.utc) - datetime.fromisoformat(row["updated_at"])
        except Exception:
            return down
        if age > timedelta(seconds=PARSER_STATUS_MAX_AGE_SECONDS):
            return down
        return bool(row["available"]), row["reason"]

    def parser_info(self) -> list[dict]:
        info = []
        for p in self.parsers.values():
            available, reason = self.availability(p.id)
            info.append({"id": p.id, "label": p.label, "available": available, "reason": reason})
        return info


def build_kb_services(
    settings: Settings,
    embed_model: BaseEmbedding | None = None,
    parsers: list[Parser] | None = None,
    threaded: bool = True,
) -> KbServices:
    if embed_model is None:
        from llama_index.embeddings.ollama import OllamaEmbedding

        # EmbeddingGemma is trained with task prefixes; they widen the gap between
        # relevant and irrelevant chunks in our retrieval checks.
        prefixes = (
            {
                "query_instruction": "task: search result | query: ",
                "text_instruction": "title: none | text: ",
            }
            if settings.embed_model.startswith("embeddinggemma")
            else {}
        )
        embed_model = OllamaEmbedding(
            model_name=settings.embed_model, base_url=settings.ollama_base_url, **prefixes
        )
    if parsers is None:
        from app.kb.parsers.lightonocr import LightOnOcrParser
        from app.kb.parsers.liteparse_parser import LiteParseParser
        from app.kb.parsers.ocr_engines import build_ocr_engine

        parsers = [
            LiteParseParser(settings.liteparse_ocr_language),
            LightOnOcrParser(build_ocr_engine(settings)),
        ]
    index = KnowledgeIndex(
        settings.chroma_dir, embed_model, settings.chroma_host, settings.chroma_port
    )
    services = KbServices(
        index=index,
        parsers={p.id: p for p in parsers},
        kb_dir=Path(settings.kb_dir),
        db_path=settings.db_path,
        external=settings.kb_worker_mode == "external",
    )
    services.worker = IngestWorker(
        settings.db_path,
        index,
        services.parsers,
        lambda kb_id, doc_id: str(services.file_path(kb_id, doc_id)),
        threaded=threaded,
        poll_seconds=settings.kb_worker_poll_seconds,
        stale_seconds=settings.kb_stale_seconds,
        heartbeat_seconds=settings.kb_heartbeat_seconds,
        kb_dir=str(services.kb_dir),
    )
    return services

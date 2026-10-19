"""One-time migration of the old Chroma knowledge bases into Milvus.

Run from the project root, with the migrate dependency group installed:

    uv run --group migrate python -m app.kb.migrate_chroma --chroma-dir data/chroma [--dry-run]

Chunks are copied as they are (no re-splitting) and re-embedded with the current
embedding service (EMBEDDER_URL). Chunks whose knowledge base or document no longer exists in SQLite are
skipped. Running it again replaces each document's chunks, so it never duplicates data.
"""

import argparse
import json
import logging
import re
import sqlite3
import sys
from pathlib import Path

from app.config import Settings
from app.db import connect
from app.kb.index import KnowledgeIndex

logger = logging.getLogger(__name__)

PAGE_SIZE = 500
_COLLECTION_RE = re.compile(r"kb_(\d+)")


def _chunk_text(document: str | None, metadata: dict) -> str:
    """LlamaIndex's Chroma store keeps the text in `documents`; fall back to the node JSON."""
    if document:
        return document
    try:
        return str(json.loads(metadata.get("_node_content", "{}")).get("text") or "")
    except (TypeError, ValueError):
        return ""


def _read_chunks(collection):
    """Yield (document_id, filename, page, text) for every chunk, in stored order."""
    offset = 0
    while True:
        batch = collection.get(
            limit=PAGE_SIZE, offset=offset, include=["documents", "metadatas"]
        )
        ids = batch["ids"]
        if not ids:
            return
        for document, metadata in zip(batch["documents"], batch["metadatas"], strict=True):
            metadata = metadata or {}
            if metadata.get("kb_document_id") is None:
                yield None, "", 0, ""
                continue
            yield (
                int(metadata["kb_document_id"]),
                str(metadata.get("filename", "")),
                int(metadata.get("page", 0)),
                _chunk_text(document, metadata),
            )
        offset += len(ids)


def migrate(
    chroma_client, db_conn: sqlite3.Connection, index: KnowledgeIndex, dry_run: bool
) -> dict[int, tuple[int, int]]:
    """Copy every `kb_<id>` collection into the index; returns kb_id -> (migrated, skipped)."""
    result: dict[int, tuple[int, int]] = {}
    for item in chroma_client.list_collections():
        name = getattr(item, "name", item)
        match = _COLLECTION_RE.fullmatch(name)
        if not match:
            continue
        kb_id = int(match.group(1))
        kb_exists = (
            db_conn.execute("SELECT 1 FROM knowledge_bases WHERE id = ?", (kb_id,)).fetchone()
            is not None
        )
        # document_id -> (filename, [(page, text)]); insertion order keeps chunk order
        docs: dict[int, tuple[str, list[tuple[int, str]]]] = {}
        skipped = 0
        live: dict[int, bool] = {}
        for doc_id, filename, page, text in _read_chunks(chroma_client.get_collection(name)):
            if doc_id is None:
                skipped += 1
                continue
            if doc_id not in live:
                live[doc_id] = kb_exists and (
                    db_conn.execute(
                        "SELECT 1 FROM kb_documents WHERE id = ? AND kb_id = ?", (doc_id, kb_id)
                    ).fetchone()
                    is not None
                )
            if not live[doc_id] or not text.strip():
                skipped += 1
                continue
            docs.setdefault(doc_id, (filename, []))[1].append((page, text))
        migrated = 0
        for doc_id, (filename, chunks) in docs.items():
            if dry_run:
                migrated += len(chunks)
            else:
                migrated += index.add_chunks(kb_id, doc_id, filename, chunks)
        result[kb_id] = (migrated, skipped)
        logger.info("知識庫 %d：遷移 %d 個 chunk，略過 %d 個", kb_id, migrated, skipped)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把舊的 Chroma 知識庫一次性遷移到 Milvus")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--chroma-dir", help="本機 Chroma 資料夾")
    source.add_argument("--chroma-host", help="Chroma 伺服器主機")
    parser.add_argument("--chroma-port", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true", help="只統計，不寫入 Milvus")
    args = parser.parse_args(argv)

    try:
        import chromadb
    except ImportError:
        print("請以 uv run --group migrate 執行遷移", file=sys.stderr)
        return 1

    from app.kb.services import build_kb_services

    logging.basicConfig(level=logging.INFO)
    if args.chroma_dir and not (Path(args.chroma_dir) / "chroma.sqlite3").is_file():
        print(f"找不到 Chroma 資料：{args.chroma_dir}", file=sys.stderr)
        return 1
    try:
        chroma_settings = chromadb.Settings(anonymized_telemetry=False)
        if args.chroma_host:
            client = chromadb.HttpClient(
                host=args.chroma_host, port=args.chroma_port, settings=chroma_settings
            )
        else:
            client = chromadb.PersistentClient(path=args.chroma_dir, settings=chroma_settings)
        settings = Settings.from_env()
        index = build_kb_services(settings, parsers=[], threaded=False).index
        conn = connect(settings.db_path)
        try:
            result = migrate(client, conn, index, args.dry_run)
        finally:
            conn.close()
    except Exception as exc:
        print(f"遷移失敗：{exc}（可安全地重新執行，遷移不會產生重複資料）", file=sys.stderr)
        return 1
    label = "（試跑，未寫入）" if args.dry_run else ""
    for kb_id, (migrated, skipped) in sorted(result.items()):
        print(f"知識庫 {kb_id}：遷移 {migrated} 個 chunk，略過 {skipped} 個{label}")
    total = sum(m for m, _ in result.values())
    print(f"完成：共遷移 {total} 個 chunk{label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

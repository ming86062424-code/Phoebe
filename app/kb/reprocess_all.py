"""Re-parse every knowledge-base document with its original parser.

Run from the project root (or, in Docker, inside the worker container):

    uv run python -m app.kb.reprocess_all [--dry-run]
    docker compose run --rm --no-deps worker python -m app.kb.reprocess_all

Every 'ready' or 'failed' document is set back to 'queued' (error and chunk_count cleared,
like a fresh upload); a running worker then re-parses and re-indexes it. Documents that are
pending, parsing or indexing are skipped, so a document a worker is busy with is never
disturbed (queue it again by running this once more when they are done).

The document's image folder is not removed here: the worker removes it before it parses
(and fails the document with a clear message if a file is in use), and deleting it from this
process could race a worker that claims the document right after it is queued.
"""

import argparse
import logging
import sqlite3
import sys

from app.config import Settings
from app.db import connect
from app.kb import store

logger = logging.getLogger(__name__)


def reprocess(conn: sqlite3.Connection, dry_run: bool) -> dict[int, int]:
    """Queue every finished document; returns kb_id -> number of documents queued."""
    return store.requeue_finished_documents(conn, dry_run=dry_run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把所有文件重新排入佇列，由 worker 依原解析器重新處理")
    parser.add_argument("--dry-run", action="store_true", help="只統計，不修改任何文件")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    try:
        settings = Settings.from_env()
        conn = connect(settings.db_path)
        try:
            counts = reprocess(conn, args.dry_run)
        finally:
            conn.close()
    except Exception as exc:
        print(f"重新處理失敗：{exc}（可安全地重新執行）", file=sys.stderr)
        return 1
    label = "（試跑，未修改）" if args.dry_run else ""
    for kb_id, count in sorted(counts.items()):
        print(f"知識庫 {kb_id}：排入 {count} 份文件{label}")
    total = sum(counts.values())
    print(f"完成：共 {total} 份文件{label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

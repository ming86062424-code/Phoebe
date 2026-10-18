import logging
import shutil
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.db import connect
from app.kb import store
from app.kb.index import KnowledgeIndex
from app.kb.parsers.base import Parser, ParseError

logger = logging.getLogger(__name__)

# how often an idle worker retries deleting files that were in use, and scans for orphans
CLEANUP_INTERVAL_SECONDS = 60


def _remove(path: Path) -> bool:
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
        return True
    except OSError as e:
        logger.warning("暫時無法刪除 %s：%s", path, e)
        return False


class _Stopped(Exception):
    """The worker was asked to shut down; the document is left for another run."""


class _Superseded(Exception):
    """The document was deleted (and maybe its id reused) while being processed."""


class IngestWorker:
    """Parses and indexes uploaded documents one at a time.

    The queue is the kb_documents table: workers atomically claim 'queued' rows, so several
    workers (or a standalone worker process) can share it. With threaded=False, enqueue()
    processes queued documents immediately (used by tests).
    """

    def __init__(
        self,
        db_path: str,
        index: KnowledgeIndex,
        parsers: dict[str, Parser],
        file_path: Callable[[int, int], str],
        threaded: bool = True,
        poll_seconds: float = 2.0,
        stale_seconds: int = 120,
        kb_dir: str = "",
        heartbeat_seconds: float = 30.0,
    ):
        self.db_path = db_path
        self.index = index
        self.parsers = parsers
        self.file_path = file_path
        self.threaded = threaded
        self.poll_seconds = poll_seconds
        self.stale_seconds = stale_seconds
        self.kb_dir = kb_dir
        self.heartbeat_seconds = heartbeat_seconds
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._pending_removal: set[Path] = set()
        self._removal_lock = threading.Lock()

    def requeue_stale(self) -> None:
        """Re-queue documents whose worker disappeared."""
        stale_before = (
            datetime.now(timezone.utc) - timedelta(seconds=self.stale_seconds)
        ).isoformat()
        conn = connect(self.db_path)
        try:
            requeued = store.requeue_stale_documents(conn, stale_before)
        finally:
            conn.close()
        if requeued:
            logger.info("重新排入逾時的文件：%s", requeued)

    def start(self) -> None:
        """Re-queue stale documents, then start the thread (or drain the queue if unthreaded)."""
        self.requeue_stale()
        if self.threaded:
            if self._thread is None:
                self._stop.clear()
                self._thread = threading.Thread(
                    target=self.run_forever, args=(self._stop,), name="kb-ingest", daemon=True
                )
                self._thread.start()
        else:
            self._drain()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._wake.set()
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.warning("解析執行緒尚未結束，保留參考以免重複啟動")
            else:
                self._thread = None

    def wake(self) -> None:
        self._wake.set()

    def enqueue(self, doc_id: int) -> None:
        """The document is already 'queued' in the database; make sure a worker looks at it."""
        if self.threaded:
            self.wake()
        else:
            self._drain()

    def _claim(self) -> int | None:
        conn = connect(self.db_path)
        try:
            return store.claim_next_document(conn)
        finally:
            conn.close()

    def _drain(self) -> None:
        while (doc_id := self._claim()) is not None:
            try:
                self.process(doc_id)
            except Exception:
                logger.exception("處理文件 %s 時發生未預期錯誤", doc_id)
        self._release_parsers()

    def run_forever(self, stop: threading.Event) -> None:
        last_scan: float | None = None
        worked = False
        while not stop.is_set():
            self._wake.clear()
            try:
                doc_id = self._claim()
            except Exception:
                logger.exception("認領文件失敗")
                doc_id = None
            if doc_id is not None:
                worked = True
                try:
                    self.process(doc_id, stop)
                except Exception:
                    logger.exception("處理文件 %s 時發生未預期錯誤", doc_id)
                continue
            if worked:
                self._release_parsers()
                worked = False
            self.cleanup()
            if last_scan is None or time.monotonic() - last_scan >= CLEANUP_INTERVAL_SECONDS:
                last_scan = time.monotonic()
                try:
                    self.requeue_stale()
                    self.cleanup_orphans()
                except Exception:
                    logger.exception("定期維護（重新排入逾時文件、清理孤立檔案）失敗")
            self._wake.wait(self.poll_seconds)

    def cleanup_orphans(self) -> int:
        """Delete document files and knowledge-base folders that have no database row."""
        root = Path(self.kb_dir)
        if not self.kb_dir or not root.is_dir():
            return 0
        # List files first, then check the database: a document row always exists
        # before its file does, so a fresh upload is never mistaken for an orphan.
        found: list[tuple[int, Path, list[tuple[int, Path]]]] = []
        for folder in root.iterdir():
            if folder.is_dir() and folder.name.isdigit():
                docs = [
                    (int(f.stem), f)
                    for f in folder.iterdir()
                    if f.is_file() and f.suffix == ".pdf" and f.stem.isdigit()
                ]
                found.append((int(folder.name), folder, docs))
        removed = 0
        conn = connect(self.db_path)
        try:
            for kb_id, folder, docs in found:
                kb_exists = conn.execute(
                    "SELECT 1 FROM knowledge_bases WHERE id = ?", (kb_id,)
                ).fetchone()
                if not kb_exists:
                    if _remove(folder):
                        removed += 1
                    continue
                for doc_id, f in docs:
                    row = conn.execute(
                        "SELECT 1 FROM kb_documents WHERE id = ? AND kb_id = ?", (doc_id, kb_id)
                    ).fetchone()
                    if row is None and _remove(f):
                        removed += 1
        finally:
            conn.close()
        if removed:
            logger.info("已清理 %d 個孤立檔案或資料夾", removed)
        return removed

    def remove_path(self, path: Path) -> None:
        """Delete a file or folder now, or later when it is no longer in use.

        On Windows a PDF that the OCR parser has open can't be deleted yet.
        """
        if not _remove(path):
            with self._removal_lock:
                self._pending_removal.add(path)

    def cleanup(self) -> None:
        """Retry deletions that failed earlier."""
        with self._removal_lock:
            pending = list(self._pending_removal)
        for path in pending:
            if _remove(path):
                with self._removal_lock:
                    self._pending_removal.discard(path)

    def _release_parsers(self) -> None:
        for parser in self.parsers.values():
            try:
                parser.release()
            except Exception:
                logger.exception("釋放解析器 %s 失敗", parser.id)

    def process(self, doc_id: int, stop: threading.Event | None = None) -> None:
        conn = connect(self.db_path)
        done = threading.Event()
        beat = threading.Thread(
            target=self._heartbeat, args=(doc_id, done), name="kb-heartbeat", daemon=True
        )
        beat.start()
        try:
            self._process(conn, doc_id, stop)
        finally:
            done.set()
            beat.join(timeout=5)
            conn.close()

    def _heartbeat(self, doc_id: int, done: threading.Event) -> None:
        """Refresh updated_at while a document is being processed, whatever the progress."""
        while not done.wait(self.heartbeat_seconds):
            try:
                conn = connect(self.db_path)
                try:
                    store.touch_document(conn, doc_id)
                finally:
                    conn.close()
            except Exception:
                logger.exception("更新文件 %s 的心跳失敗", doc_id)

    def _process(self, conn, doc_id: int, stop: threading.Event | None = None) -> None:
        doc = store.get_document(conn, doc_id)
        if doc is None or doc["status"] != "parsing":  # must have been claimed
            return
        parser = self.parsers.get(doc["parser"])
        if parser is None:
            store.update_document(conn, doc_id, status="failed", error=f"未知的解析器 {doc['parser']}")
            return

        # The row we started with; anything written later must still belong to it.
        identity = doc["created_at"]

        def check_current() -> None:
            current = store.get_document(conn, doc_id)
            if current is None or current["created_at"] != identity:
                raise _Superseded()

        def on_progress(done: int, total: int) -> None:
            if stop is not None and stop.is_set():
                raise _Stopped()  # finish after the current page; the stale check requeues it
            check_current()  # also stops a long OCR run once the document is deleted
            store.update_document(conn, doc_id, progress_done=done, progress_total=total)

        indexed = False
        try:
            pages = parser.parse(self.file_path(doc["kb_id"], doc_id), on_progress)
            check_current()
            store.update_document(conn, doc_id, status="indexing")
            # Clear vectors from any earlier, interrupted attempt before writing new ones.
            self.index.delete_document(doc["kb_id"], doc_id)
            chunks = self.index.add_document(doc["kb_id"], doc_id, doc["filename"], pages)
            indexed = True
            check_current()
            if chunks == 0:
                store.update_document(
                    conn, doc_id, status="failed", error="沒有擷取到任何文字，掃描檔請改用 LightOnOCR"
                )
                return
            store.update_document(conn, doc_id, status="ready", chunk_count=chunks)
        except _Stopped:
            logger.info("worker 停止中，文件 %s 留待下次處理", doc_id)
        except _Superseded:
            if indexed:
                self.index.delete_document(doc["kb_id"], doc_id)
        except ParseError as e:
            self._fail(conn, doc_id, identity, str(e))
        except Exception as e:
            logger.exception("文件 %s 處理失敗", doc_id)
            self._fail(conn, doc_id, identity, f"處理失敗：{e}")

    def _fail(self, conn, doc_id: int, identity: str, message: str) -> None:
        current = store.get_document(conn, doc_id)
        if current is not None and current["created_at"] == identity:
            store.update_document(conn, doc_id, status="failed", error=message)

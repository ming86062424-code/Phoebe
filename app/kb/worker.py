import logging
import queue
import shutil
import threading
from collections.abc import Callable
from pathlib import Path

from app.db import connect
from app.kb import store
from app.kb.index import KnowledgeIndex
from app.kb.parsers.base import Parser, ParseError

logger = logging.getLogger(__name__)

# how often an idle worker retries deleting files that were in use
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


class _Superseded(Exception):
    """The document was deleted (and maybe its id reused) while being processed."""


class IngestWorker:
    """Parses and indexes uploaded documents one at a time.

    With threaded=False, enqueue() processes the document immediately (used by tests).
    """

    def __init__(
        self,
        db_path: str,
        index: KnowledgeIndex,
        parsers: dict[str, Parser],
        file_path: Callable[[int, int], str],
        threaded: bool = True,
    ):
        self.db_path = db_path
        self.index = index
        self.parsers = parsers
        self.file_path = file_path
        self.threaded = threaded
        self._queue: queue.Queue[int | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._pending_removal: set[Path] = set()
        self._removal_lock = threading.Lock()

    def start(self) -> None:
        """Re-queue documents left unfinished by a previous run, then start the thread."""
        conn = connect(self.db_path)
        try:
            pending = [d["id"] for d in store.list_unfinished_documents(conn)]
            for doc_id in pending:
                store.update_document(conn, doc_id, status="queued", progress_done=0)
        finally:
            conn.close()
        if self.threaded and self._thread is None:
            self._thread = threading.Thread(target=self._run, name="kb-ingest", daemon=True)
            self._thread.start()
        for doc_id in pending:
            self.enqueue(doc_id)

    def stop(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=5)
            self._thread = None

    def enqueue(self, doc_id: int) -> None:
        if self.threaded:
            self._queue.put(doc_id)
        else:
            self.process(doc_id)
            self._release_parsers()

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

    def _run(self) -> None:
        while True:
            try:
                doc_id = self._queue.get(timeout=CLEANUP_INTERVAL_SECONDS)
            except queue.Empty:
                self.cleanup()
                continue
            if doc_id is None:
                return
            try:
                self.process(doc_id)
            except Exception:
                logger.exception("處理文件 %s 時發生未預期錯誤", doc_id)
            if self._queue.empty():
                self._release_parsers()
                self.cleanup()

    def _release_parsers(self) -> None:
        for parser in self.parsers.values():
            try:
                parser.release()
            except Exception:
                logger.exception("釋放解析器 %s 失敗", parser.id)

    def process(self, doc_id: int) -> None:
        conn = connect(self.db_path)
        try:
            self._process(conn, doc_id)
        finally:
            conn.close()

    def _process(self, conn, doc_id: int) -> None:
        doc = store.get_document(conn, doc_id)
        if doc is None or doc["status"] != "queued":
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
            check_current()  # also stops a long OCR run once the document is deleted
            store.update_document(conn, doc_id, progress_done=done, progress_total=total)

        store.update_document(conn, doc_id, status="parsing", error=None, progress_done=0)
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

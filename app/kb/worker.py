import logging
import queue
import threading
from collections.abc import Callable

from app.db import connect
from app.kb import store
from app.kb.index import KnowledgeIndex
from app.kb.parsers.base import Parser, ParseError

logger = logging.getLogger(__name__)


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

    def _run(self) -> None:
        while True:
            doc_id = self._queue.get()
            if doc_id is None:
                return
            try:
                self.process(doc_id)
            except Exception:
                logger.exception("處理文件 %s 時發生未預期錯誤", doc_id)
            if self._queue.empty():
                self._release_parsers()

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
            self._fail(conn, doc_id, f"未知的解析器 {doc['parser']}")
            return

        def on_progress(done: int, total: int) -> None:
            store.update_document(conn, doc_id, progress_done=done, progress_total=total)

        store.update_document(conn, doc_id, status="parsing", error=None, progress_done=0)
        try:
            pages = parser.parse(self.file_path(doc["kb_id"], doc_id), on_progress)
            if store.get_document(conn, doc_id) is None:
                return  # deleted while parsing
            store.update_document(conn, doc_id, status="indexing")
            # Clear vectors from any earlier, interrupted attempt before writing new ones.
            self.index.delete_document(doc["kb_id"], doc_id)
            chunks = self.index.add_document(doc["kb_id"], doc_id, doc["filename"], pages)
        except ParseError as e:
            self._fail(conn, doc_id, str(e))
            return
        except Exception as e:
            logger.exception("文件 %s 處理失敗", doc_id)
            self._fail(conn, doc_id, f"處理失敗：{e}")
            return
        if store.get_document(conn, doc_id) is None:
            self.index.delete_document(doc["kb_id"], doc_id)  # deleted while indexing
            return
        if chunks == 0:
            self._fail(conn, doc_id, "沒有擷取到任何文字，掃描檔請改用 LightOnOCR")
            return
        store.update_document(conn, doc_id, status="ready", chunk_count=chunks)

    def _fail(self, conn, doc_id: int, message: str) -> None:
        if store.get_document(conn, doc_id) is not None:
            store.update_document(conn, doc_id, status="failed", error=message)

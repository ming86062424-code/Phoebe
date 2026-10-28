import logging
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.db import connect, now_iso
from app.kb import store
from app.kb.filetypes import UPLOAD_EXTENSIONS
from app.kb.index import KnowledgeIndex
from app.kb.parsers.base import Parser, ParseError
from app.monitor.store import record_event

logger = logging.getLogger(__name__)

# how often an idle ingester retries deleting files that were in use, and scans for orphans
CLEANUP_INTERVAL_SECONDS = 60
# upload temp files older than this are leftovers from an interrupted upload
UPLOAD_TMP_MAX_AGE_SECONDS = 3600
# how often the ingester republishes which parsers it can run (the web process reads it)
STATUS_PUBLISH_SECONDS = 15
STALE_FOLDER_ERROR = "無法清除上次處理留下的圖片（檔案可能正被使用），請稍後重試"


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
    """The ingester was asked to shut down; the document is left for another run."""


class _Superseded(Exception):
    """The document was deleted (and maybe its id reused) while being processed."""


class Ingester:
    """Parses and indexes uploaded documents, up to `concurrency` at a time.

    The queue is the kb_documents table: ingesters atomically claim 'queued' rows, so several
    ingesters (or a standalone ingest process) can share it. With threaded=False, enqueue()
    processes queued documents immediately and one at a time (used by tests).
    """

    def __init__(
        self,
        db_path: str,
        index: KnowledgeIndex,
        parsers: dict[str, Parser],
        file_path: Callable[[int, int, str], str],
        threaded: bool = True,
        poll_seconds: float = 2.0,
        stale_seconds: int = 120,
        kb_dir: str = "",
        heartbeat_seconds: float = 30.0,
        doc_dir: Callable[[int, int], Path] | None = None,
        concurrency: int = 1,
    ):
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.db_path = db_path
        self.index = index
        self.parsers = parsers
        self.file_path = file_path  # (kb_id, doc_id, filename) -> uploaded file
        # (kb_id, doc_id) -> the folder the parser writes the document's images into
        self.doc_dir = doc_dir or (lambda kb_id, doc_id: Path(kb_dir) / str(kb_id) / str(doc_id))
        self.threaded = threaded
        self.poll_seconds = poll_seconds
        self.stale_seconds = stale_seconds
        self.kb_dir = kb_dir
        self.heartbeat_seconds = heartbeat_seconds
        self.concurrency = concurrency
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # Wakes the dispatcher: wake() and every finished document set _woken and notify.
        # A flag rather than a bare notify, so a wake-up that comes while the dispatcher is
        # busy claiming is not lost. _in_flight is the number of documents being processed.
        self._cond = threading.Condition()
        self._woken = False
        self._in_flight = 0
        self._status_thread: threading.Thread | None = None
        self._status_stop = threading.Event()
        self._pending_removal: set[Path] = set()
        self._removal_lock = threading.Lock()
        self._waiting_for: tuple[str, ...] = ()  # parsers whose documents are held back

    def requeue_stale(self) -> None:
        """Re-queue documents whose ingester disappeared."""
        stale_before = (
            datetime.now(timezone.utc) - timedelta(seconds=self.stale_seconds)
        ).isoformat()
        conn = connect(self.db_path)
        try:
            requeued = store.requeue_stale_documents(conn, stale_before)
            abandoned = store.fail_stale_pending_documents(
                conn, stale_before, "上傳未完成，請重新上傳"
            )
        finally:
            conn.close()
        if requeued:
            logger.info("重新排入逾時的文件：%s", requeued)
        if abandoned:
            logger.info("標記為失敗的未完成上傳：%s", abandoned)

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
            self.start_status_publisher()
        else:
            self._drain()

    def publish_parser_status(self) -> None:
        """Record in the database which parsers can run here, for the web process to read."""
        rows = []
        for parser in self.parsers.values():
            try:
                available, reason = parser.availability()
            except Exception as e:
                logger.exception("檢查解析器 %s 失敗", parser.id)
                available, reason = False, f"檢查失敗：{e}"
            rows.append((parser.id, int(bool(available)), reason or "", now_iso()))
        conn = connect(self.db_path)
        try:
            conn.executemany(
                "INSERT INTO parser_status (parser_id, available, reason, updated_at)"
                " VALUES (?, ?, ?, ?) ON CONFLICT(parser_id) DO UPDATE SET"
                " available = excluded.available, reason = excluded.reason,"
                " updated_at = excluded.updated_at",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def start_status_publisher(self, interval: float = STATUS_PUBLISH_SECONDS) -> None:
        """Publish parser status from its own thread, so a long OCR run can't make it stale."""
        if self._status_thread is not None:
            return
        self._status_stop.clear()
        self._status_thread = threading.Thread(
            target=self._publish_loop,
            args=(self._status_stop, interval),
            name="kb-parser-status",
            daemon=True,
        )
        self._status_thread.start()

    def stop_status_publisher(self) -> None:
        thread = self._status_thread
        if thread is None:
            return
        self._status_stop.set()
        thread.join(timeout=5)
        if not thread.is_alive():
            self._status_thread = None

    def _publish_loop(self, stop: threading.Event, interval: float) -> None:
        while True:
            try:
                self.publish_parser_status()
            except Exception:
                logger.exception("發佈解析器狀態失敗")
            if stop.wait(interval):
                return

    def stop(self) -> None:
        self.stop_status_publisher()
        if self._thread is not None:
            self._stop.set()
            self.wake()
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.warning("解析執行緒尚未結束，保留參考以免重複啟動")
            else:
                self._thread = None

    def wake(self) -> None:
        # Condition wraps an RLock, so this is safe from a signal handler that interrupts
        # the dispatcher on the main thread while it holds the lock (ingest_main).
        with self._cond:
            self._woken = True
            self._cond.notify_all()

    def enqueue(self, doc_id: int) -> None:
        """The document is already 'queued' in the database; make sure the ingester looks at it."""
        if self.threaded:
            self.wake()
        else:
            self._drain()

    def _claim(self) -> int | None:
        """Claim the oldest queued document whose parser can run now.

        A document whose parser is unavailable (e.g. the vLLM OCR service is still loading
        after a restart) stays queued instead of failing, and is picked up once the parser
        is back. Only the parsers of queued documents are checked, so an idle ingester never
        polls the OCR service. Unknown parsers are still claimed (and fail clearly).
        """
        conn = connect(self.db_path)
        try:
            queued = store.queued_parsers(conn)
            if not queued:
                return None
            waiting = tuple(
                sorted(p for p in queued if p in self.parsers and not self._available(p))
            )
            if waiting != self._waiting_for:
                if waiting:
                    logger.info("解析器 %s 目前無法使用，相關文件先保留在佇列中", "、".join(waiting))
                self._waiting_for = waiting
            return store.claim_next_document(conn, waiting)
        finally:
            conn.close()

    def _available(self, parser_id: str) -> bool:
        try:
            return bool(self.parsers[parser_id].availability()[0])
        except Exception:
            logger.warning("檢查解析器 %s 是否可用時失敗，先視為無法使用", parser_id, exc_info=True)
            return False

    def _drain(self) -> None:
        while (doc_id := self._claim()) is not None:
            try:
                self.process(doc_id)
            except Exception:
                logger.exception("處理文件 %s 時發生未預期錯誤", doc_id)
        self._release_parsers()

    def run_forever(self, stop: threading.Event) -> None:
        """The dispatcher: claims documents and hands them to a pool of `concurrency` threads.

        Only this thread claims documents (and so reads and writes _waiting_for) and runs the
        maintenance. Parsers are released only once no document is in flight and none can
        be claimed, so a model another thread is using is never unloaded. On stop it claims
        nothing more and waits for the pool; documents in flight stop at their next page.
        """
        last_scan: float | None = None
        worked = False
        with self._cond:
            self._in_flight = 0
        pool = ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="kb-ingest")
        try:
            while not stop.is_set():
                with self._cond:
                    # cleared before looking at the queue: anything that happens from here
                    # on sets it again, so the wait below can't miss it
                    self._woken = False
                    in_flight = self._in_flight
                doc_id = None
                if in_flight < self.concurrency:
                    try:
                        doc_id = self._claim()
                    except Exception:
                        logger.exception("認領文件失敗")
                if doc_id is not None:
                    worked = True
                    self._submit(pool, doc_id, stop)
                    continue
                if in_flight == 0:
                    if worked:
                        self._release_parsers()
                        worked = False
                    self.cleanup()
                    if (
                        last_scan is None
                        or time.monotonic() - last_scan >= CLEANUP_INTERVAL_SECONDS
                    ):
                        last_scan = time.monotonic()
                        try:
                            self.requeue_stale()
                            self.cleanup_orphans()
                        except Exception:
                            logger.exception("定期維護（重新排入逾時文件、清理孤立檔案）失敗")
                with self._cond:
                    self._cond.wait_for(
                        lambda: self._woken or stop.is_set(), timeout=self.poll_seconds
                    )
        finally:
            pool.shutdown(wait=True)

    def _submit(self, pool: ThreadPoolExecutor, doc_id: int, stop: threading.Event) -> None:
        with self._cond:
            self._in_flight += 1
        try:
            pool.submit(self._run_one, doc_id, stop)
        except BaseException:
            self._finished()  # never ran: the document stays 'parsing' until requeued as stale
            raise

    def _run_one(self, doc_id: int, stop: threading.Event) -> None:
        try:
            self.process(doc_id, stop)
        except Exception:
            logger.exception("處理文件 %s 時發生未預期錯誤", doc_id)
        finally:
            self._finished()

    def _finished(self) -> None:
        with self._cond:
            self._in_flight -= 1
            self._woken = True
            self._cond.notify_all()

    def cleanup_orphans(self) -> int:
        """Delete document files and folders, and knowledge-base folders, with no database row."""
        root = Path(self.kb_dir)
        if not self.kb_dir or not root.is_dir():
            return 0
        # List files first, then check the database: a document row always exists
        # before its file does, so a fresh upload is never mistaken for an orphan.
        found: list[tuple[int, Path, list[tuple[int, Path]]]] = []
        for folder in root.iterdir():
            if folder.is_dir() and folder.name.isdigit():
                docs = [
                    (int(f.stem if f.is_file() else f.name), f)
                    for f in folder.iterdir()
                    if (f.is_file() and f.suffix.lower() in UPLOAD_EXTENSIONS and f.stem.isdigit())
                    or (f.is_dir() and f.name.isdigit())
                ]
                found.append((int(folder.name), folder, docs))
        removed = 0
        tmp_cutoff = time.time() - UPLOAD_TMP_MAX_AGE_SECONDS
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
                for tmp in folder.glob("upload-*.tmp"):
                    try:
                        old = tmp.is_file() and tmp.stat().st_mtime < tmp_cutoff
                    except OSError:
                        continue
                    if old and _remove(tmp):
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
        # monitoring event (docs/monitoring.md): outcome and seconds per phase
        started, phases = time.time(), {}
        outcome: dict = {"status": "error"}
        try:
            out_dir = self.doc_dir(doc["kb_id"], doc_id)
            # images from an earlier attempt must not linger next to the new ones
            if out_dir.exists() and not _remove(out_dir):
                outcome["error"] = STALE_FOLDER_ERROR
                self._fail(conn, doc_id, identity, STALE_FOLDER_ERROR)
                return
            out_dir.mkdir(parents=True, exist_ok=True)
            path = self.file_path(doc["kb_id"], doc_id, doc["filename"])
            parse_started = time.perf_counter()
            parsed = parser.parse(path, out_dir, on_progress)
            phases["parse"] = time.perf_counter() - parse_started
            outcome["pages"] = len(parsed)
            check_current()
            store.update_document(conn, doc_id, status="indexing")
            # Clear vectors from any earlier, interrupted attempt before writing new ones.
            self.index.delete_document(doc["kb_id"], doc_id)
            chunks = self.index.add_document(
                doc["kb_id"], doc_id, doc["filename"], parsed, image_dir=out_dir, timings=phases
            )
            indexed = True
            outcome["chunks"] = chunks
            check_current()
            if chunks == 0:
                error = (
                    "檔案中沒有任何文字"
                    if parser.id == "text"
                    else "沒有擷取到任何文字，掃描檔請改用 LightOnOCR"
                )
                store.update_document(conn, doc_id, status="failed", error=error)
                outcome["error"] = error
                return
            store.update_document(conn, doc_id, status="ready", chunk_count=chunks)
            outcome["status"] = "ok"
        except _Stopped:
            logger.info("ingest 停止中，文件 %s 留待下次處理", doc_id)
            outcome["status"] = "stopped"
        except _Superseded:
            outcome["status"] = "deleted"
            if indexed:
                self.index.delete_document(doc["kb_id"], doc_id)
            self._remove_leftovers(conn, doc["kb_id"], doc_id)
        except ParseError as e:
            outcome["error"] = str(e)
            self._fail(conn, doc_id, identity, str(e))
        except Exception as e:
            logger.exception("文件 %s 處理失敗", doc_id)
            outcome["error"] = f"處理失敗：{e}"
            self._fail(conn, doc_id, identity, outcome["error"])
        finally:
            status = outcome.pop("status")
            record_event(
                self.db_path,
                "document",
                status,
                started,
                time.time(),
                doc["filename"],
                phases,
                {
                    "kb_id": doc["kb_id"],
                    "document_id": doc_id,
                    "parser": parser.id,
                    **outcome,
                },
            )

    def _remove_leftovers(self, conn, kb_id: int, doc_id: int) -> None:
        """After a document was deleted mid-run: drop the folder this run (re)created, and
        the knowledge base's folder too when the knowledge base is gone."""
        if store.get_document(conn, doc_id) is not None:
            return  # the id belongs to a new document now; its folder is not ours to delete
        self.remove_path(self.doc_dir(kb_id, doc_id))
        if conn.execute("SELECT 1 FROM knowledge_bases WHERE id = ?", (kb_id,)).fetchone() is None:
            self.remove_path(self.doc_dir(kb_id, doc_id).parent)

    def _fail(self, conn, doc_id: int, identity: str, message: str) -> None:
        current = store.get_document(conn, doc_id)
        if current is not None and current["created_at"] == identity:
            store.update_document(conn, doc_id, status="failed", error=message)

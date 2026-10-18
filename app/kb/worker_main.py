"""Standalone ingest worker: python -m app.kb.worker_main"""

import logging
import signal
import threading

from app.config import Settings
from app.db import connect, init_db
from app.kb.services import build_kb_services

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    conn = connect(settings.db_path)
    try:
        init_db(conn)
    finally:
        conn.close()
    services = build_kb_services(settings)
    worker = services.worker
    stop = threading.Event()

    def request_stop(signum, frame) -> None:
        logger.info("收到停止訊號，結束中")
        stop.set()
        worker.wake()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    # The loop runs on the main thread, so don't call start() (it would spawn a second one).
    worker.requeue_stale()
    worker.start_status_publisher()
    logger.info("解析 worker 已啟動")
    try:
        worker.run_forever(stop)
    finally:
        worker.stop_status_publisher()
    logger.info("解析 worker 已停止")


if __name__ == "__main__":
    main()

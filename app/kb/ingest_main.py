"""Standalone ingest service: python -m app.kb.ingest_main"""

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
    ingester = services.ingester
    stop = threading.Event()

    def request_stop(signum, frame) -> None:
        logger.info("收到停止訊號，結束中")
        stop.set()
        ingester.wake()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    # The loop runs on the main thread, so don't call start() (it would spawn a second one).
    ingester.requeue_stale()
    ingester.start_status_publisher()
    logger.info("ingest 已啟動，同時處理 %d 份文件", settings.ingest_concurrency)
    try:
        ingester.run_forever(stop)
    finally:
        ingester.stop_status_publisher()
    logger.info("ingest 已停止")


if __name__ == "__main__":
    main()

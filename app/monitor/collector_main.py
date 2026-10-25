"""Entry point of the monitor service: python -m app.monitor.collector_main"""

import logging
import signal
import threading

from app.config import Settings
from app.monitor.collector import Collector


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    # every sample is ~10 Docker API calls: httpx's per-request INFO lines would flood the log
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = Settings.from_env()
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    logging.getLogger("app.monitor").info(
        "monitor 已啟動，每 %.0f 秒取樣一次", settings.monitor_interval_seconds
    )
    Collector(
        settings.db_path,
        settings.monitor_interval_seconds,
        settings.monitor_resource_retention_hours,
        settings.monitor_event_retention_days,
    ).run_forever(stop)


if __name__ == "__main__":
    main()

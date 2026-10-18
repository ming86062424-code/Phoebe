"""Manual check for the LightOnOCR parser.

Usage: uv run --group ocr python -m scripts.ocr_smoke path/to/file.pdf
"""

import sys
import time

from app.config import Settings
from app.kb.parsers.lightonocr import LightOnOcrParser
from app.kb.parsers.ocr_engines import build_ocr_engine


def main() -> None:
    settings = Settings.from_env()
    parser = LightOnOcrParser(build_ocr_engine(settings))
    ok, reason = parser.availability()
    if not ok:
        sys.exit(reason)
    start = time.time()
    pages = parser.parse(sys.argv[1], lambda done, total: print(f"頁 {done}/{total}", flush=True))
    for p in pages:
        print(f"\n===== 第 {p.page} 頁 =====\n{p.text}")
    print(f"\n耗時 {time.time() - start:.1f} 秒")
    parser.release()


if __name__ == "__main__":
    main()

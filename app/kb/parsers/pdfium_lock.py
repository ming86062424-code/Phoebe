"""One process-wide lock for pypdfium2.

pdfium is not thread-safe: no two threads may call it at the same time, not even on
different documents. Every pypdfium2 object call (open, len, page access, get_size, render,
to_pil, close) holds this lock; pages and bitmaps are closed explicitly inside it so no
finalizer frees a native handle on another thread. It is reentrant so a caller may hold it
across several calls.

LiteParse does not need it. liteparse 2.16.0 wheels bundle their own pdfium next to the
native extension (liteparse/pdfium.dll + _liteparse.pyd on Windows, liteparse/libpdfium.so
+ _liteparse.abi3.so on manylinux), so its pdfium state is separate from pypdfium2's.
tests/test_parsers.py runs LiteParse parse+screenshot concurrently with locked pypdfium2
renders to confirm.
"""

import threading

PDFIUM_LOCK = threading.RLock()

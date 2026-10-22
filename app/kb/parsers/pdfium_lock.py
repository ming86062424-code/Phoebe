"""One process-wide lock for pypdfium2.

pdfium is not thread-safe: no two threads may call it at the same time, not even on
different documents. Every pypdfium2 call (open, len, page access, render, close) holds
this lock. It is reentrant so a caller may hold it across several calls.
"""

import threading

PDFIUM_LOCK = threading.RLock()

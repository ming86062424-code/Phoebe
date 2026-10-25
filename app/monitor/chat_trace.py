"""Timing of one chat reply, stored as a "chat" monitoring event when the reply ends.

Phases (seconds): embed and search (knowledge base retrieval), images (loading the
attached pictures), first_token (from calling the model to its first delta, which may be
thinking), thinking (first delta to the first answer text, when the model thinks first)
and answer (first answer text to the end). Message text is never recorded.
"""

import time

from app.monitor.store import record_event


class ChatTrace:
    def __init__(self, app_db_path: str, title: str, details: dict):
        self.app_db_path = app_db_path
        self.title = title
        self.details = dict(details)
        self.phases: dict[str, float] = {}
        self.started = time.time()
        self._llm_started: float | None = None
        self._first_delta: float | None = None
        self._first_content: float | None = None
        self._finished = False

    def llm_started(self) -> None:
        self._llm_started = time.perf_counter()

    def delta(self, kind: str) -> None:
        """Called for every streamed delta ("thinking" or "content")."""
        now = time.perf_counter()
        if self._first_delta is None:
            self._first_delta = now
            if self._llm_started is not None:
                self.phases["first_token"] = now - self._llm_started
        if kind == "content" and self._first_content is None:
            self._first_content = now
            if now > self._first_delta:
                self.phases["thinking"] = now - self._first_delta

    def finish(self, status: str, error: str | None = None) -> None:
        """Record the event once; later calls (e.g. the stream's cleanup) do nothing."""
        if self._finished:
            return
        self._finished = True
        if self._first_content is not None:
            self.phases["answer"] = time.perf_counter() - self._first_content
        details = {**self.details, **({"error": error} if error else {})}
        record_event(
            self.app_db_path, "chat", status, self.started, time.time(), self.title,
            self.phases, details,
        )

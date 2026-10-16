from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

# Called after each page with (pages_done, pages_total).
ProgressCallback = Callable[[int, int], None]


class ParseError(Exception):
    pass


@dataclass(frozen=True)
class PageText:
    page: int  # 1-based
    text: str


class Parser(Protocol):
    id: str
    label: str

    def availability(self) -> tuple[bool, str]:
        """(available, reason shown to the user when unavailable)."""
        ...

    def parse(self, pdf_path: str, on_progress: ProgressCallback) -> list[PageText]: ...

    def release(self) -> None:
        """Free heavy resources (e.g. GPU memory) once the queue is idle."""
        ...

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

# Called after each page with (pages_done, pages_total).
ProgressCallback = Callable[[int, int], None]

ChunkKind = Literal["text", "page_image"]


class ParseError(Exception):
    pass


@dataclass(frozen=True)
class ImageRef:
    """A PNG the parser wrote into the document's folder (`kb_dir/<kb>/<doc>/<name>`)."""

    name: str
    width: int
    height: int


@dataclass(frozen=True)
class ParsedChunk:
    """One chunk to index: a PDF page, an image file, or a piece of a text file.

    kind "text" is embedded from its text; kind "page_image" from `page_image` plus text.
    """

    page: int  # 1-based for PDFs and images, 0 for text files
    text: str
    images: list[ImageRef] = field(default_factory=list)
    page_image: ImageRef | None = None
    kind: ChunkKind = "text"


class Parser(Protocol):
    id: str
    label: str

    def availability(self) -> tuple[bool, str]:
        """(available, reason shown to the user when unavailable)."""
        ...

    def parse(
        self, path: str, out_dir: Path, on_progress: ProgressCallback
    ) -> list[ParsedChunk]:
        """Parse the uploaded file; images the chunks refer to are written into out_dir."""
        ...

    def release(self) -> None:
        """Free heavy resources (e.g. GPU memory) once the queue is idle."""
        ...

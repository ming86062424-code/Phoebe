import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

# Called after each page with (pages_done, pages_total).
ProgressCallback = Callable[[int, int], None]

ChunkKind = Literal["text", "page_image"]

# Names of the PNGs a parser writes: `page_{n}.png` (LiteParse page image) and
# `img_{page}_{i}.png` (LightOnOCR crop). ASCII and bounded (at most 9 digits per number,
# 28 characters), so a reference always fits Milvus' 256-byte `page_image` field.
IMAGE_NAME = re.compile(r"^(page|img)_[0-9]{1,9}(_[0-9]{1,9})?\.png$")
MAX_IMAGE_SIDE = 999_999


class ParseError(Exception):
    pass


@dataclass(frozen=True)
class ImageRef:
    """A PNG the parser wrote into the document's folder (`kb_dir/<kb>/<doc>/<name>`)."""

    name: str
    width: int
    height: int

    def __post_init__(self):
        if not isinstance(self.name, str) or not IMAGE_NAME.fullmatch(self.name):
            raise ValueError(f"invalid image name: {self.name!r}")
        for side in (self.width, self.height):
            if not isinstance(side, int) or not 0 < side <= MAX_IMAGE_SIDE:
                raise ValueError(f"invalid image size: {self.width}x{self.height}")


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

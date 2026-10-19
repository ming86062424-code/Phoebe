"""Plain text (.txt) and Markdown (.md) files: split into chunks of at most 1500 characters."""

import re
from pathlib import Path

from app.kb.parsers.base import ParsedChunk, ParseError, ProgressCallback

MAX_CHARS = 1500
UNREADABLE = "無法讀取檔案內容"

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_HEADING = re.compile(r"^#{1,6}(\s|$)")


def decode_text(data: bytes) -> str:
    """UTF-8 with or without a BOM; anything else raises UnicodeDecodeError."""
    return data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")


def _paragraphs(text: str) -> list[str]:
    return [p.strip("\n") for p in _PARAGRAPH_BREAK.split(text) if p.strip()]


def _merge(paragraphs: list[str], limit: int) -> list[str]:
    """Join paragraphs with blank lines up to `limit` characters; hard-split longer ones."""
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        while len(para) > limit:
            chunks.append(para[:limit])
            para = para[limit:]
        current = para
    if current:
        chunks.append(current)
    return chunks


def split_text(text: str, limit: int = MAX_CHARS) -> list[str]:
    return _merge(_paragraphs(text.replace("\r\n", "\n")), limit)


def _sections(text: str) -> list[str]:
    """Split before every heading line (`#`), ignoring lines inside code fences."""
    sections: list[list[str]] = [[]]
    fence = ""  # the opening marker while inside a code fence
    for line in text.split("\n"):
        marker = _FENCE.match(line)
        if fence:
            # closes only on the same character, at least as long, with nothing after it
            if (
                marker
                and marker.group(1)[0] == fence[0]
                and len(marker.group(1)) >= len(fence)
                and not marker.group(2).strip()
            ):
                fence = ""
        elif marker:
            fence = marker.group(1)
        elif _HEADING.match(line) and sections[-1]:
            sections.append([])
        sections[-1].append(line)
    return ["\n".join(lines).strip() for lines in sections if "\n".join(lines).strip()]


def split_markdown(text: str, limit: int = MAX_CHARS) -> list[str]:
    chunks: list[str] = []
    for section in _sections(text.replace("\r\n", "\n")):
        if len(section) <= limit:
            chunks.append(section)
            continue
        first, _, rest = section.partition("\n")
        if _HEADING.match(first) and len(first) <= limit // 3:
            # every piece of a long section repeats its heading
            budget = limit - len(first) - 2
            chunks.extend(f"{first}\n\n{piece}" for piece in _merge(_paragraphs(rest), budget))
        else:
            chunks.extend(_merge(_paragraphs(section), limit))
    return chunks


class TextParser:
    """Chosen automatically for .txt and .md uploads; not selectable by the user."""

    id = "text"
    label = "文字檔"

    def availability(self) -> tuple[bool, str]:
        return True, ""

    def parse(
        self, path: str, out_dir: Path, on_progress: ProgressCallback
    ) -> list[ParsedChunk]:
        try:
            text = decode_text(Path(path).read_bytes())
        except (OSError, UnicodeDecodeError) as e:
            raise ParseError(UNREADABLE) from e
        split = split_markdown if Path(path).suffix.lower() == ".md" else split_text
        chunks = [ParsedChunk(0, piece) for piece in split(text)]
        on_progress(1, 1)
        return chunks

    def release(self) -> None:
        pass

"""LightOnOCR in its official grounding mode: layout blocks with boxes, image blocks cropped.

The output format and the block parser follow the official client
(github.com/lightonai/LightOnOCR, lightonocr/grounding.py): one marker per layout block,

    ![label](x1,y1,x2,y2) block text ...

where the text runs until the next marker (it may start on the next line and span several
lines), the box is normalised to 0-1000 on both axes of the image sent to the model, and a
trailing "+" on the label (`text+`) marks a block continuing the previous one.
"""

import logging
import math
import re
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from app.kb.filetypes import IMAGE_EXTENSIONS
from app.kb.parsers.base import ImageRef, ParsedChunk, ParseError, ProgressCallback
from app.kb.parsers.images import MAX_PIXELS, fit, open_image, save_png
from app.kb.parsers.ocr_engines import OcrEngine
from app.kb.parsers.pdfium_lock import PDFIUM_LOCK

logger = logging.getLogger(__name__)

# Model card: "Render PDFs at 400 DPI to images using a target longest dimension of 2048px"
RENDER_DPI = 400
MAX_EDGE = 2048

# The official marker pattern (lightonocr/grounding.py MARKER).
MARKER = re.compile(
    r"!\[([A-Za-z_][\w-]*?)(\+?)\]\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)[ \t]*"
)
# Page chrome left out of the chunk: header, footer and page number (and the header/footer
# variants the official HEADER_FOOTER set lists). Footnotes are kept: they are content.
DROPPED_LABELS = frozenset(
    {"header", "footer", "page_number", "page_header", "page_footer", "header_image", "footer_image"}
)
IMAGE_LABEL = "image"
# Crops kept per page: even at the longest names and sizes ImageRef allows, 100 references
# fit Milvus' 8192-byte `images` field, so the list never drops an image the text refers to.
# Image blocks past the cap keep their description as plain text, without a reference.
MAX_CROPS_PER_PAGE = 100


@dataclass(frozen=True)
class Block:
    label: str
    bbox: tuple[int, int, int, int] | None  # x1, y1, x2, y2 on 0-1000; None for unmarked text
    text: str
    continues: bool = False  # the label carried a trailing "+"


def parse_grounding(text: str) -> list[Block]:
    """Split grounding output into blocks; each marker owns the text up to the next marker.

    Text before the first marker (or output without any marker) becomes an unboxed "text"
    block, and a malformed marker stays in the text of the block it sits in, so no output
    is lost.
    """
    marks = list(MARKER.finditer(text))
    blocks = []
    lead = text[: marks[0].start() if marks else len(text)].strip()
    if lead:
        blocks.append(Block("text", None, lead))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        blocks.append(
            Block(
                label=m.group(1),
                bbox=(int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6))),
                text=text[m.end() : end].strip(),
                continues=m.group(2) == "+",
            )
        )
    return blocks


def to_pixels(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    """A 0-1000 box as a pixel box on a width x height image, ordered and clamped to it.

    Rounded outwards so the crop never cuts into the block.
    """
    x1, x2 = sorted((bbox[0], bbox[2]))
    y1, y2 = sorted((bbox[1], bbox[3]))

    def clamp(v: float, hi: int) -> int:
        return min(max(int(v), 0), hi)

    return (
        clamp(math.floor(x1 * width / 1000), width),
        clamp(math.floor(y1 * height / 1000), height),
        clamp(math.ceil(x2 * width / 1000), width),
        clamp(math.ceil(y2 * height / 1000), height),
    )


def page_markdown(blocks: list[Block], image, page: int, out_dir: Path) -> tuple[str, list[ImageRef]]:
    """The page as markdown in model order, without page chrome; image blocks are cropped
    from `image` (the exact image the model saw) into `img_{page}_{i}.png`."""
    parts: list[str] = []
    refs: list[ImageRef] = []
    prev_kept = False
    skipped = 0
    for block in blocks:
        if block.label in DROPPED_LABELS:
            prev_kept = False
            continue
        content = block.text
        if block.label == IMAGE_LABEL and block.bbox is not None:
            box = to_pixels(block.bbox, image.width, image.height)
            if len(refs) >= MAX_CROPS_PER_PAGE:
                skipped += 1
            elif box[2] > box[0] and box[3] > box[1]:
                i = len(refs) + 1
                ref = save_png(image.crop(box), out_dir, f"img_{page}_{i}.png")
                refs.append(ref)
                content = f"![圖片 {i}](img:{ref.name}) {block.text}".rstrip()
        if not content:
            prev_kept = False
            continue
        # official to_markdown: a continuation block joins the previous block's paragraph
        if block.continues and prev_kept:
            parts[-1] = f"{parts[-1]} {content}"
        else:
            parts.append(content)
        prev_kept = True
    if skipped:
        logger.warning(
            "第 %d 頁有 %d 張圖片超過上限 %d，只保留描述文字", page, skipped, MAX_CROPS_PER_PAGE
        )
    return "\n\n".join(parts), refs


def _render_scale(width_pt: float, height_pt: float) -> float:
    """400 DPI, lowered only when the render would exceed the pixel cap."""
    scale = RENDER_DPI / 72
    pixels = width_pt * height_pt * scale * scale
    if width_pt > 0 and height_pt > 0 and pixels > MAX_PIXELS:
        scale *= math.sqrt(MAX_PIXELS / pixels) * 0.999
    return scale


class LightOnOcrParser:
    id = "lightonocr"
    label = "LightOnOCR-3（GPU OCR，適合掃描檔與複雜版面）"

    def __init__(self, engine: OcrEngine, concurrency: int = 1):
        self.engine = engine
        # pages of one document sent to the engine at the same time
        self.concurrency = max(1, concurrency)

    def availability(self) -> tuple[bool, str]:
        return self.engine.availability()

    @staticmethod
    def _ocr_result(page: int, ocr: Callable[[], str]) -> str:
        """The engine's output, with engine failures reported as this page's failure."""
        try:
            return ocr()
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(f"第 {page} 頁 OCR 失敗：{e}") from e

    @staticmethod
    def _chunk(raw: str, image, page: int, out_dir: Path) -> ParsedChunk:
        text, refs = page_markdown(parse_grounding(raw), image, page, out_dir)
        return ParsedChunk(page, text, images=refs)

    def parse(
        self, path: str, out_dir: Path, on_progress: ProgressCallback
    ) -> list[ParsedChunk]:
        if Path(path).suffix.lower() in IMAGE_EXTENSIONS:
            on_progress(0, 1)
            # an image file is one page, sized like a rendered page (never upscaled)
            image = fit(open_image(path), MAX_EDGE)
            raw = self._ocr_result(1, lambda: self.engine.ocr(image))
            chunk = self._chunk(raw, image, 1, out_dir)
            on_progress(1, 1)
            return [chunk]

        import pypdfium2 as pdfium

        # pdfium is not thread-safe: hold the lock only around its calls, never around OCR.
        try:
            with PDFIUM_LOCK:
                pdf = pdfium.PdfDocument(path)
        except Exception as e:
            raise ParseError(f"LightOnOCR 初始化失敗：{e}") from e
        # Pipeline: this thread renders pages one at a time and sends each to the pool; at
        # most `concurrency` pages are in flight (so at most that many rendered images are
        # alive). Results are taken oldest first, so cropping, saving and progress happen
        # here, in page order. The pool belongs to this parse: documents never share it.
        pool = ThreadPoolExecutor(
            max_workers=self.concurrency, thread_name_prefix="lightonocr"
        )
        in_flight: deque[tuple[int, object, Future[str]]] = deque()
        try:
            with PDFIUM_LOCK:
                total = len(pdf)
            on_progress(0, total)
            pages: list[ParsedChunk] = []

            def finish_oldest() -> None:
                page_no, image, future = in_flight.popleft()
                raw = self._ocr_result(page_no, future.result)
                pages.append(self._chunk(raw, image, page_no, out_dir))
                on_progress(len(pages), total)

            for i in range(total):
                if len(in_flight) >= self.concurrency:
                    finish_oldest()
                with PDFIUM_LOCK:
                    page = pdf[i]
                    try:
                        # rendering as the model card does: 400 DPI, then a longest edge of 2048 px
                        scale = _render_scale(*page.get_size())
                        bitmap = page.render(scale=scale)
                        try:
                            # to_pil() stays inside the lock: the bitmap belongs to pdfium
                            image = bitmap.to_pil().convert("RGB")
                        finally:
                            bitmap.close()
                    finally:
                        # free the native handle here, not in a finalizer on another thread
                        page.close()
                image.thumbnail((MAX_EDGE, MAX_EDGE))
                in_flight.append((i + 1, image, pool.submit(self.engine.ocr, image)))
                del image
            while in_flight:
                finish_oldest()
            return pages
        finally:
            # On a failed page or a stop raised by on_progress: drop the pages not yet
            # started and wait for the running ones, so no OCR outlives this call.
            for _, _, future in in_flight:
                future.cancel()
            in_flight.clear()
            pool.shutdown(wait=True, cancel_futures=True)
            with PDFIUM_LOCK:
                pdf.close()

    def release(self) -> None:
        self.engine.release()

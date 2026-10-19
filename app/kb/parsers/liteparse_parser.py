import io
import logging
import math
from pathlib import Path

from app.kb.filetypes import IMAGE_EXTENSIONS
from app.kb.parsers.base import ParsedChunk, ParseError, ProgressCallback
from app.kb.parsers.images import MAX_PIXELS, check_pixels, fit, open_image, save_png, to_rgb

logger = logging.getLogger(__name__)

# Each page is rendered at 400 DPI, then downscaled to a longest edge of 1536 px; that page
# image (with the page's text) is what gets embedded.
RENDER_DPI = 400
PAGE_IMAGE_EDGE = 1536
POINTS_PER_INCH = 72


def render_dpi(width_pt: float, height_pt: float) -> float:
    """400 DPI, or less for a page so large that 400 DPI would exceed the pixel cap."""
    if width_pt <= 0 or height_pt <= 0:
        return RENDER_DPI
    pixels = (width_pt * RENDER_DPI / POINTS_PER_INCH) * (height_pt * RENDER_DPI / POINTS_PER_INCH)
    if pixels <= MAX_PIXELS:
        return RENDER_DPI
    # floor to a whole DPI so rounding can't push the render back over the cap
    return float(math.floor(RENDER_DPI * math.sqrt(MAX_PIXELS / pixels)))


class LiteParseParser:
    id = "liteparse"
    label = "LiteParse（快速，適合文字型 PDF）"

    def __init__(self, ocr_language: str):
        self.ocr_language = ocr_language

    def availability(self) -> tuple[bool, str]:
        return True, ""

    def _text_parser(self):
        from liteparse import LiteParse

        return LiteParse(output_format="markdown", quiet=True, ocr_language=self.ocr_language)

    def parse(
        self, path: str, out_dir: Path, on_progress: ProgressCallback
    ) -> list[ParsedChunk]:
        if Path(path).suffix.lower() in IMAGE_EXTENSIONS:
            return self._parse_image(path, out_dir, on_progress)
        return self._parse_pdf(path, out_dir, on_progress)

    def _parse_pdf(
        self, path: str, out_dir: Path, on_progress: ProgressCallback
    ) -> list[ParsedChunk]:
        from liteparse import LiteParse

        try:
            result = self._text_parser().parse(path)
        except Exception as e:
            raise ParseError(f"LiteParse 解析失敗：{e}") from e
        total = len(result.pages)
        on_progress(0, total)
        # Page images come from LiteParse's own renderer (LiteParse.screenshot, which renders
        # at the instance's `dpi`), one page per call so only one page is in memory at a time.
        renderers: dict[float, LiteParse] = {}
        chunks = []
        for done, page in enumerate(result.pages, start=1):
            dpi = render_dpi(page.width, page.height)
            if dpi not in renderers:
                renderers[dpi] = LiteParse(dpi=dpi, quiet=True, ocr_enabled=False)
            try:
                shots = renderers[dpi].screenshot(path, page_numbers=[page.page_num])
                if not shots:
                    raise ParseError("沒有產生頁面圖片")
                image = _decode_png(shots[0].image_bytes)
            except Exception as e:
                raise ParseError(f"LiteParse 第 {page.page_num} 頁轉成圖片失敗：{e}") from e
            ref = save_png(fit(image, PAGE_IMAGE_EDGE), out_dir, f"page_{page.page_num}.png")
            chunks.append(
                ParsedChunk(
                    page.page_num,
                    (page.markdown or page.text or "").strip(),
                    page_image=ref,
                    kind="page_image",
                )
            )
            on_progress(done, total)
        return chunks

    def _parse_image(
        self, path: str, out_dir: Path, on_progress: ProgressCallback
    ) -> list[ParsedChunk]:
        """An image file is one page: the upright RGB image itself is the page image."""
        on_progress(0, 1)
        image = open_image(path)  # format, pixel cap, full decode, EXIF transpose
        ref = save_png(fit(image, PAGE_IMAGE_EDGE), out_dir, "page_1.png")
        try:  # LiteParse OCRs images (Tesseract); without text the page is still embedded
            pages = self._text_parser().parse(path).pages
            text = (pages[0].markdown or pages[0].text or "").strip() if pages else ""
        except Exception as e:
            logger.warning("LiteParse 無法取得圖片 %s 的文字：%s", path, e)
            text = ""
        on_progress(1, 1)
        return [ParsedChunk(1, text, page_image=ref, kind="page_image")]

    def release(self) -> None:
        pass


def _decode_png(data: bytes):
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        check_pixels(image.width, image.height)
        image.load()
        return to_rgb(image)

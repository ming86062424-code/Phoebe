import io
import logging
import math
import re
import tempfile
from pathlib import Path

from app.kb.filetypes import IMAGE_EXTENSIONS
from app.kb.parsers.base import ImageRef, ParsedChunk, ParseError, ProgressCallback
from app.kb.parsers.images import MAX_PIXELS, check_pixels, fit, open_image, save_png, to_rgb
from app.kb.parsers.pdfium_lock import PDFIUM_LOCK

logger = logging.getLogger(__name__)

# Each page is rendered at 400 DPI, then downscaled to a longest edge of 1536 px; that page
# image (with the page's text) is what gets embedded.
RENDER_DPI = 400
PAGE_IMAGE_EDGE = 1536
POINTS_PER_INCH = 72
# LiteParse stops after `max_pages` (1000 by default); the cap is lifted to the PDF's real
# page count so a long PDF is never cut short silently.
LITEPARSE_MAX_PAGES = 1000
# Images embedded in the PDF are extracted (LiteParse `extract_images`) and saved as
# img_{page}_{i}.png, like LightOnOCR's crops; LiteParse's markdown refers to them as
# ![alt](img_p{page}_{i}.jpg).
MAX_IMAGES_PER_PAGE = 100  # keeps the images JSON within Milvus' 8192 bytes
MIN_IMAGE_SIDE = 16  # smaller pictures are spacers or bullets
_IMAGE_LINK = re.compile(r"!\[([^\]]*)\]\(([^)\s]*)\)")


def link_images(
    markdown: str, page: int, src_dir: Path | None, out_dir: Path
) -> tuple[str, list[ImageRef]]:
    """Save the images LiteParse extracted into `src_dir` as img_{page}_{i}.png in
    `out_dir`, and point the markdown at them (`![圖片 i](img:name) alt`).

    A placeholder without a usable image (missing, unreadable, tiny, or over the per-page
    cap) is replaced by its alt text, so no dangling reference is left. Images of this
    page that the markdown does not mention (LiteParse leaves some out) are appended.
    """
    refs: list[ImageRef] = []
    used: set[str] = set()

    def take(name: str, alt: str) -> str | None:
        """The markdown for the image file `name`, or None when it is not usable."""
        if src_dir is None or len(refs) >= MAX_IMAGES_PER_PAGE:
            return None
        # only the file name: a link can never reach outside LiteParse's folder
        source = src_dir / Path(name).name
        used.add(source.name)
        if not source.is_file():
            return None
        try:
            image = open_image(source)  # format, pixel cap, EXIF, RGB
        except ParseError:
            logger.warning("LiteParse 抽出的圖片無法讀取，略過：%s", source.name)
            return None
        if min(image.size) < MIN_IMAGE_SIDE:
            return None
        i = len(refs) + 1
        refs.append(save_png(fit(image, PAGE_IMAGE_EDGE), out_dir, f"img_{page}_{i}.png"))
        return f"![圖片 {i}](img:{refs[-1].name}) {alt}".rstrip()

    def replace(match: re.Match) -> str:
        alt = match.group(1).strip()
        return take(match.group(2), alt) or alt

    text = _IMAGE_LINK.sub(replace, markdown)
    extra = []
    if src_dir is not None:
        for source in _page_images(src_dir, page):
            if source.name not in used:
                linked = take(source.name, "")
                if linked:
                    extra.append(linked)
    if extra:
        text = "\n\n".join([text.rstrip(), *extra]) if text.strip() else "\n\n".join(extra)
    return text, refs


def _page_images(src_dir: Path, page: int) -> list[Path]:
    """LiteParse's extracted images of one page (img_p{page}_{k}.*), in k order."""
    pattern = re.compile(rf"^img_p{page}_(\d+)\.\w+$")
    found = [(int(m.group(1)), p) for p in src_dir.iterdir() if (m := pattern.match(p.name))]
    return [p for _, p in sorted(found)]


def pdf_page_count(path: str) -> int:
    """The PDF's page count (0 when pdfium can't tell; LiteParse then reports the error)."""
    import pypdfium2 as pdfium

    try:
        with PDFIUM_LOCK:
            pdf = pdfium.PdfDocument(path)
    except Exception:
        return 0
    try:
        with PDFIUM_LOCK:
            return len(pdf)
    finally:
        with PDFIUM_LOCK:
            pdf.close()


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

    def _text_parser(self, max_pages: int = LITEPARSE_MAX_PAGES, image_dir: Path | None = None):
        from liteparse import LiteParse

        images = {"extract_images": True, "image_output_dir": str(image_dir)} if image_dir else {}
        return LiteParse(
            output_format="markdown",
            quiet=True,
            ocr_language=self.ocr_language,
            max_pages=max_pages,
            **images,
        )

    def parse(
        self, path: str, out_dir: Path, on_progress: ProgressCallback
    ) -> list[ParsedChunk]:
        if Path(path).suffix.lower() in IMAGE_EXTENSIONS:
            return self._parse_image(path, out_dir, on_progress)
        return self._parse_pdf(path, out_dir, on_progress)

    def _parse_pdf(
        self, path: str, out_dir: Path, on_progress: ProgressCallback
    ) -> list[ParsedChunk]:
        # LiteParse writes the PDF's embedded images to a scratch folder; link_images
        # copies the usable ones into out_dir under the names the image API serves
        with tempfile.TemporaryDirectory(prefix="liteparse-") as scratch:
            return self._parse_pdf_with_images(path, out_dir, Path(scratch), on_progress)

    def _parse_pdf_with_images(
        self, path: str, out_dir: Path, image_dir: Path, on_progress: ProgressCallback
    ) -> list[ParsedChunk]:
        from liteparse import LiteParse

        count = pdf_page_count(path)
        try:
            result = self._text_parser(max(count, LITEPARSE_MAX_PAGES), image_dir).parse(path)
        except Exception as e:
            raise ParseError(f"LiteParse 解析失敗：{e}") from e
        total = len(result.pages)
        if count and total < count:  # never index a silently shortened document
            raise ParseError(f"LiteParse 只解析了 {total} 頁（共 {count} 頁）")
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
            text, images = link_images(
                page.markdown or page.text or "", page.page_num, image_dir, out_dir
            )
            chunks.append(
                ParsedChunk(
                    page.page_num, text.strip(), images, page_image=ref, kind="page_image"
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
        # LiteParse OCRs images (Tesseract) on the upright, normalised page image; without
        # text the page is still embedded
        try:
            pages = self._text_parser().parse(out_dir / ref.name).pages
            text = (pages[0].markdown or pages[0].text or "") if pages else ""
            # the image file is already the page image: drop any image placeholder
            text = link_images(text, 1, None, out_dir)[0].strip()
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

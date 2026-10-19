from pathlib import Path

from app.kb.parsers.base import ParsedChunk, ParseError, ProgressCallback


class LiteParseParser:
    id = "liteparse"
    label = "LiteParse（快速，適合文字型 PDF）"

    def __init__(self, ocr_language: str):
        self.ocr_language = ocr_language

    def availability(self) -> tuple[bool, str]:
        return True, ""

    def parse(
        self, path: str, out_dir: Path, on_progress: ProgressCallback
    ) -> list[ParsedChunk]:
        from liteparse import LiteParse

        try:
            result = LiteParse(
                output_format="markdown", quiet=True, ocr_language=self.ocr_language
            ).parse(path)
        except Exception as e:
            raise ParseError(f"LiteParse 解析失敗：{e}") from e
        pages = [
            ParsedChunk(page.page_num, (page.markdown or page.text or "").strip())
            for page in result.pages
        ]
        on_progress(len(pages), len(pages))
        return pages

    def release(self) -> None:
        pass

from app.kb.parsers.base import PageText, ParseError, ProgressCallback


class LiteParseParser:
    id = "liteparse"
    label = "LiteParse（快速，適合文字型 PDF）"

    def __init__(self, ocr_language: str):
        self.ocr_language = ocr_language

    def availability(self) -> tuple[bool, str]:
        return True, ""

    def parse(self, pdf_path: str, on_progress: ProgressCallback) -> list[PageText]:
        from liteparse import LiteParse

        try:
            result = LiteParse(
                output_format="markdown", quiet=True, ocr_language=self.ocr_language
            ).parse(pdf_path)
        except Exception as e:
            raise ParseError(f"LiteParse 解析失敗：{e}") from e
        pages = [
            PageText(page.page_num, (page.markdown or page.text or "").strip())
            for page in result.pages
        ]
        on_progress(len(pages), len(pages))
        return pages

    def release(self) -> None:
        pass

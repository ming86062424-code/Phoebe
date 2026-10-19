from app.kb.parsers.base import PageText, ParseError, ProgressCallback
from app.kb.parsers.ocr_engines import OcrEngine

RENDER_DPI = 400
MAX_EDGE = 2048


class LightOnOcrParser:
    id = "lightonocr"
    label = "LightOnOCR-3（GPU OCR，適合掃描檔與複雜版面）"

    def __init__(self, engine: OcrEngine):
        self.engine = engine

    def availability(self) -> tuple[bool, str]:
        return self.engine.availability()

    def parse(self, pdf_path: str, on_progress: ProgressCallback) -> list[PageText]:
        import pypdfium2 as pdfium

        try:
            pdf = pdfium.PdfDocument(pdf_path)
        except Exception as e:
            raise ParseError(f"LightOnOCR 初始化失敗：{e}") from e
        try:
            total = len(pdf)
            on_progress(0, total)
            pages: list[PageText] = []
            for i in range(total):
                page = pdf[i]
                # rendering exactly as the model card does
                image = page.render(scale=RENDER_DPI / 72).to_pil().convert("RGB")
                image.thumbnail((MAX_EDGE, MAX_EDGE))
                try:
                    text = self.engine.ocr(image)
                except ParseError:
                    raise
                except Exception as e:
                    raise ParseError(f"第 {i + 1} 頁 OCR 失敗：{e}") from e
                pages.append(PageText(i + 1, text))
                on_progress(i + 1, total)
            return pages
        finally:
            pdf.close()

    def release(self) -> None:
        self.engine.release()

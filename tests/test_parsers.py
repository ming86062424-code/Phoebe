from app.kb.parsers.liteparse_parser import LiteParseParser
from tests.pdf_factory import make_pdf


def test_liteparse_returns_text_per_page(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(make_pdf(["Hello alpha page one", "Beta page two"]))
    progress = []
    pages = LiteParseParser("eng").parse(str(path), lambda d, t: progress.append((d, t)))
    assert [(p.page, p.text) for p in pages] == [
        (1, "Hello alpha page one"),
        (2, "Beta page two"),
    ]
    assert progress[-1] == (2, 2)


def test_liteparse_raises_parse_error_on_garbage(tmp_path):
    import pytest

    from app.kb.parsers.base import ParseError

    path = tmp_path / "bad.pdf"
    path.write_bytes(b"%PDF-1.4 not really")
    with pytest.raises(ParseError):
        LiteParseParser("eng").parse(str(path), lambda d, t: None)


def test_lightonocr_uses_greedy_decoding():
    """Sampling makes OCR text differ between runs; greedy decoding is reproducible."""
    torch = __import__("pytest").importorskip("torch")

    from app.kb.parsers.lightonocr import LightOnOcrParser

    class FakeInputs(dict):
        def to(self, device):
            return self

    class FakeProcessor:
        def apply_chat_template(self, *args, **kwargs):
            return FakeInputs(input_ids=torch.zeros((1, 3), dtype=torch.long))

        def decode(self, tokens, skip_special_tokens):
            return "文字"

    class FakeModel:
        device = "cpu"

        def __init__(self):
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return torch.zeros((1, 5), dtype=torch.long)

    parser = LightOnOcrParser("model", "http://ollama")
    parser._model, parser._processor = FakeModel(), FakeProcessor()
    assert parser._ocr_image(object()) == "文字"
    assert parser._model.kwargs["do_sample"] is False
    assert "temperature" not in parser._model.kwargs

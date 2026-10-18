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


def test_lightonocr_generate_kwargs():
    """Official model-card sampling (temperature 0.1). The model config ships use_cache=false, which recomputes the whole prompt per token."""
    torch = __import__("pytest").importorskip("torch")

    from app.kb.parsers.ocr_engines import TransformersOcrEngine

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

    parser = TransformersOcrEngine("model", "http://ollama")
    parser._model, parser._processor = FakeModel(), FakeProcessor()
    assert parser.ocr(object()) == "文字"
    assert parser._model.kwargs["do_sample"] is True
    assert parser._model.kwargs["temperature"] == 0.1
    assert parser._model.kwargs["use_cache"] is True


def test_lightonocr_parser_uses_engine_per_page(tmp_path):
    from app.kb.parsers.lightonocr import LightOnOcrParser

    class FakeEngine:
        def __init__(self):
            self.calls = 0

        def availability(self):
            return True, ""

        def ocr(self, image):
            self.calls += 1
            return f"頁{self.calls}"

        def release(self):
            pass

    path = tmp_path / "a.pdf"
    path.write_bytes(make_pdf(["one", "two", "three"]))
    engine = FakeEngine()
    parser = LightOnOcrParser(engine)
    progress = []
    pages = parser.parse(str(path), lambda d, t: progress.append((d, t)))
    assert engine.calls == 3
    assert [(p.page, p.text) for p in pages] == [(1, "頁1"), (2, "頁2"), (3, "頁3")]
    assert progress == [(0, 3), (1, 3), (2, 3), (3, 3)]
    assert parser.availability() == (True, "")

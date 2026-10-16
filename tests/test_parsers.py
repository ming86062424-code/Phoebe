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

import pytest

from app.kb.parsers.liteparse_parser import LiteParseParser
from tests.pdf_factory import make_pdf


def _image_file(path, size=(300, 200), fmt=None, orientation=None, mode="RGB"):
    from PIL import Image

    image = Image.new(mode, size, "white" if mode == "RGB" else (255, 0, 0, 0))
    kwargs = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        kwargs["exif"] = exif
    image.save(path, format=fmt, **kwargs)
    return path


def test_liteparse_returns_page_images_with_text(tmp_path):
    from PIL import Image

    from app.kb.parsers.base import ImageRef

    path = tmp_path / "a.pdf"
    path.write_bytes(make_pdf(["Hello alpha page one", "Beta page two"]))
    out = tmp_path / "out"
    out.mkdir()
    progress = []
    pages = LiteParseParser("eng").parse(str(path), out, lambda d, t: progress.append((d, t)))
    assert all(p.kind == "page_image" for p in pages)
    assert [(p.page, p.text) for p in pages] == [
        (1, "Hello alpha page one"),
        (2, "Beta page two"),
    ]
    # Letter page at 400 DPI is 3400 x 4400 px, downscaled to a longest edge of 1536
    assert [p.page_image for p in pages] == [
        ImageRef("page_1.png", 1187, 1536),
        ImageRef("page_2.png", 1187, 1536),
    ]
    for p in pages:
        assert p.images == []
        with Image.open(out / p.page_image.name) as image:
            assert (image.format, image.mode, image.size) == ("PNG", "RGB", (1187, 1536))
    assert progress == [(0, 2), (1, 2), (2, 2)]


def test_liteparse_image_file_is_one_upright_page(tmp_path):
    from PIL import Image

    from app.kb.parsers.base import ImageRef

    # EXIF orientation 6: stored landscape, displayed rotated to portrait
    path = _image_file(tmp_path / "photo.jpg", size=(3000, 2000), fmt="JPEG", orientation=6)
    out = tmp_path / "out"
    out.mkdir()
    progress = []
    pages = LiteParseParser("eng").parse(str(path), out, lambda d, t: progress.append((d, t)))
    assert len(pages) == 1
    page = pages[0]
    assert (page.page, page.kind) == (1, "page_image")
    assert page.page_image == ImageRef("page_1.png", 1024, 1536)
    with Image.open(out / "page_1.png") as image:
        assert (image.mode, image.size) == ("RGB", (1024, 1536))
    assert progress[-1] == (1, 1)


def test_liteparse_flattens_transparent_png(tmp_path):
    from PIL import Image

    path = _image_file(tmp_path / "t.png", size=(40, 30), fmt="PNG", mode="RGBA")
    pages = LiteParseParser("eng").parse(str(path), tmp_path, lambda d, t: None)
    with Image.open(tmp_path / "page_1.png") as image:
        assert image.mode == "RGB" and image.getpixel((0, 0)) == (255, 255, 255)
    assert pages[0].page_image.width == 40  # small images are never upscaled


def test_parsers_reject_images_whose_content_does_not_match(tmp_path):
    import pytest

    from app.kb.parsers.base import ParseError
    from app.kb.parsers.lightonocr import LightOnOcrParser

    path = _image_file(tmp_path / "fake.png", fmt="JPEG")  # JPEG bytes named .png
    with pytest.raises(ParseError, match="無法讀取檔案內容"):
        LiteParseParser("eng").parse(str(path), tmp_path, lambda d, t: None)
    with pytest.raises(ParseError, match="無法讀取檔案內容"):
        LightOnOcrParser(FakeEngine()).parse(str(path), tmp_path, lambda d, t: None)


def test_parsers_reject_images_over_the_pixel_cap(tmp_path, monkeypatch):
    import pytest

    from app.kb.parsers import images
    from app.kb.parsers.base import ParseError

    monkeypatch.setattr(images, "MAX_PIXELS", 100)
    path = _image_file(tmp_path / "big.png", size=(20, 20), fmt="PNG")
    with pytest.raises(ParseError, match="無法讀取檔案內容"):
        LiteParseParser("eng").parse(str(path), tmp_path, lambda d, t: None)


def test_liteparse_raises_parse_error_on_garbage(tmp_path):
    import pytest

    from app.kb.parsers.base import ParseError

    path = tmp_path / "bad.pdf"
    path.write_bytes(b"%PDF-1.4 not really")
    with pytest.raises(ParseError):
        LiteParseParser("eng").parse(str(path), tmp_path, lambda d, t: None)


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




def test_transformers_engine_requests_grounding(monkeypatch):
    """Official usage: one user message, the image then the text "grounding".
    Runs without torch: a stand-in module provides what ocr() touches."""
    import contextlib
    import sys
    import types

    import numpy as np

    from app.kb.parsers.ocr_engines import TransformersOcrEngine

    fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    seen = {}

    class FakeInputs(dict):
        def to(self, device):
            return self

    class FakeProcessor:
        def apply_chat_template(self, conversation, **kwargs):
            seen["conversation"], seen["kwargs"] = conversation, kwargs
            return FakeInputs(input_ids=np.zeros((1, 3)))

        def decode(self, tokens, skip_special_tokens):
            seen["new_tokens"] = len(tokens)
            return "![text](1,2,3,4) x"

    class FakeModel:
        device = "cpu"

        def generate(self, **kwargs):
            seen["generate"] = kwargs
            return np.zeros((1, 5))

    engine = TransformersOcrEngine("model", "http://ollama")
    engine._model, engine._processor = FakeModel(), FakeProcessor()
    image = object()
    assert engine.ocr(image) == "![text](1,2,3,4) x"
    assert seen["conversation"] == [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": "grounding"}],
        }
    ]
    assert seen["kwargs"]["enable_thinking"] is False
    assert seen["new_tokens"] == 2
    assert seen["generate"]["temperature"] == 0.1 and seen["generate"]["do_sample"] is True


# ---- LightOnOCR grounding ----

# Shaped like the official grounding output (github.com/lightonai/LightOnOCR README and the
# model card): a marker per block, text on the same line or the lines after it.
PAGE_OUTPUT = """![header](80,20,920,45) ACME 年度報告
![title](80,70,920,140)
# Company report 2025

![text](80,175,920,225)
Revenue rose 20%.
![text+](80,230,920,250) It is still rising.

![image](80,285,460,545)
A solar-powered factory.

![chart](540,285,920,545)
<table>
  <tr><th>Year</th><th>Revenue</th></tr>
</table>
![image](100,600,300,700)
![footer](80,950,920,970) Confidential
![page_number](491,975,505,990) 3"""


class FakeEngine:
    def __init__(self, outputs=None):
        self.outputs = outputs
        self.images = []

    def availability(self):
        return True, ""

    def ocr(self, image):
        self.images.append(image.copy())
        if self.outputs is None:
            return f"頁{len(self.images)}"
        return self.outputs[len(self.images) - 1]

    def release(self):
        pass


def test_parse_grounding_official_readme_example():
    from app.kb.parsers.lightonocr import Block, parse_grounding

    raw = (
        "![image](309,90,691,516) Model architecture diagram showing NaViT, Projection\n"
        "![caption](400,555,591,566) Figure 1: Model architecture.\n"
        "![title](113,608,263,623) ## 2.2 LightOnOCR-1\n"
        "![text](112,640,882,698) The first version of LightOnOCR [13]\n"
        "![page_number](491,935,505,950) 3"
    )
    assert parse_grounding(raw) == [
        Block("image", (309, 90, 691, 516), "Model architecture diagram showing NaViT, Projection"),
        Block("caption", (400, 555, 591, 566), "Figure 1: Model architecture."),
        Block("title", (113, 608, 263, 623), "## 2.2 LightOnOCR-1"),
        Block("text", (112, 640, 882, 698), "The first version of LightOnOCR [13]"),
        Block("page_number", (491, 935, 505, 950), "3"),
    ]


def test_parse_grounding_multiline_continuation_and_stray_text():
    from app.kb.parsers.lightonocr import Block, parse_grounding

    raw = (
        "before any marker\n"
        "![table](10,20,30,40)\n<table>\n<tr><td>1</td></tr>\n</table>\n"
        "![text+](0,0,10,10) more\n"
        "![image](1,2,3) broken marker"
    )
    assert parse_grounding(raw) == [
        Block("text", None, "before any marker"),
        Block("table", (10, 20, 30, 40), "<table>\n<tr><td>1</td></tr>\n</table>"),
        Block("text", (0, 0, 10, 10), "more\n![image](1,2,3) broken marker", continues=True),
    ]
    assert parse_grounding("") == []
    assert parse_grounding("plain text only") == [Block("text", None, "plain text only")]


def test_lightonocr_grounding_page_markdown_and_crops(tmp_path):
    from PIL import Image

    from app.kb.parsers.base import ImageRef
    from app.kb.parsers.lightonocr import LightOnOcrParser

    path = tmp_path / "a.pdf"
    path.write_bytes(make_pdf(["one", "two"]))
    out = tmp_path / "out"
    out.mkdir()
    engine = FakeEngine([PAGE_OUTPUT, "![text](0,0,1000,1000) 第二頁"])
    progress = []
    pages = LightOnOcrParser(engine).parse(str(path), out, lambda d, t: progress.append((d, t)))
    assert progress == [(0, 2), (1, 2), (2, 2)]
    assert [(p.page, p.kind) for p in pages] == [(1, "text"), (2, "text")]

    sent = engine.images[0]
    assert max(sent.size) == 2048  # the model card's page size
    w, h = sent.size

    def box(x1, y1, x2, y2):
        import math

        return (
            math.floor(x1 * w / 1000),
            math.floor(y1 * h / 1000),
            math.ceil(x2 * w / 1000),
            math.ceil(y2 * h / 1000),
        )

    b1, b2 = box(80, 285, 460, 545), box(100, 600, 300, 700)
    first = pages[0]
    assert first.images == [
        ImageRef("img_1_1.png", b1[2] - b1[0], b1[3] - b1[1]),
        ImageRef("img_1_2.png", b2[2] - b2[0], b2[3] - b2[1]),
    ]
    for ref, b in zip(first.images, (b1, b2)):
        with Image.open(out / ref.name) as crop:
            assert crop.size == (ref.width, ref.height)
            assert crop.tobytes() == sent.crop(b).tobytes()
    assert first.text == (
        "# Company report 2025\n\n"
        "Revenue rose 20%. It is still rising.\n\n"
        "![圖片 1](img:img_1_1.png) A solar-powered factory.\n\n"
        "<table>\n  <tr><th>Year</th><th>Revenue</th></tr>\n</table>\n\n"
        "![圖片 2](img:img_1_2.png)"
    )
    for dropped in ("ACME", "Confidential", "\n3"):
        assert dropped not in first.text
    assert (pages[1].text, pages[1].images) == ("第二頁", [])


def test_lightonocr_image_file_crops_are_clamped_to_the_image(tmp_path):
    from PIL import Image, ImageDraw

    from app.kb.parsers.base import ImageRef
    from app.kb.parsers.lightonocr import LightOnOcrParser

    image = Image.new("RGB", (1000, 500), "white")
    ImageDraw.Draw(image).rectangle((100, 100, 299, 199), fill=(255, 0, 0))
    path = tmp_path / "scan.png"
    image.save(path)
    out = tmp_path / "out"
    out.mkdir()
    raw = (
        "![image](100,200,300,400) 紅色方塊\n"
        "![image](900,800,1200,1300) 超出邊界\n"
        "![image](500,500,500,600) 零面積\n"
        "![image](700,0,600,100)"
    )
    engine = FakeEngine([raw])
    progress = []
    pages = LightOnOcrParser(engine).parse(str(path), out, lambda d, t: progress.append((d, t)))
    assert progress == [(0, 1), (1, 1)]
    assert engine.images[0].size == (1000, 500)  # never upscaled
    assert len(pages) == 1 and pages[0].page == 1
    assert pages[0].images == [
        ImageRef("img_1_1.png", 200, 100),
        ImageRef("img_1_2.png", 100, 100),
        ImageRef("img_1_3.png", 100, 50),
    ]
    with Image.open(out / "img_1_1.png") as crop:
        assert crop.getcolors() == [(200 * 100, (255, 0, 0))]
    assert pages[0].text == (
        "![圖片 1](img:img_1_1.png) 紅色方塊\n\n"
        "![圖片 2](img:img_1_2.png) 超出邊界\n\n"
        "零面積\n\n"
        "![圖片 3](img:img_1_3.png)"
    )


def test_lightonocr_keeps_output_without_markers(tmp_path):
    from app.kb.parsers.lightonocr import LightOnOcrParser

    path = tmp_path / "a.pdf"
    path.write_bytes(make_pdf(["one", "two", "three"]))
    engine = FakeEngine()
    parser = LightOnOcrParser(engine)
    pages = parser.parse(str(path), tmp_path, lambda d, t: None)
    assert [(p.page, p.text, p.kind) for p in pages] == [
        (1, "頁1", "text"), (2, "頁2", "text"), (3, "頁3", "text"),
    ]
    assert parser.availability() == (True, "")


def test_image_ref_names_are_short_and_whitelisted():
    import pytest

    from app.kb.parsers.base import ImageRef

    ImageRef("page_123456789.png", 999_999, 1)
    ImageRef("img_12_3.png", 1, 1)
    for name in ("../app.db", "page_1.jpg", "page_1234567890.png", "圖_1.png", "page_1.png/x"):
        with pytest.raises(ValueError):
            ImageRef(name, 1, 1)
    for size in ((0, 1), (1, 1_000_000)):
        with pytest.raises(ValueError):
            ImageRef("page_1.png", *size)


def test_liteparse_lowers_dpi_only_for_huge_pages():
    from app.kb.parsers import images
    from app.kb.parsers.liteparse_parser import render_dpi

    assert render_dpi(612, 792) == 400  # Letter
    dpi = render_dpi(2384, 3370)  # A0 would be ~248 MP at 400 DPI
    assert dpi < 400
    assert (2384 * dpi / 72) * (3370 * dpi / 72) <= images.MAX_PIXELS


# ---- fix round 1 ----


def _mpo(path):
    from PIL import Image

    first = Image.new("RGB", (64, 48), (200, 30, 30))
    second = Image.new("RGB", (64, 48), (30, 30, 200))
    first.save(path, format="MPO", save_all=True, append_images=[second])
    return path


def test_mpo_photos_count_as_jpeg(tmp_path):
    from PIL import Image

    from app.kb.parsers.images import open_image
    from app.kb.parsers.lightonocr import LightOnOcrParser

    path = _mpo(tmp_path / "phone.jpg")
    with Image.open(path) as raw:
        assert raw.format == "MPO"
    assert open_image(path).size == (64, 48)
    out = tmp_path / "out"
    out.mkdir()
    lite = LiteParseParser("eng").parse(str(path), out, lambda d, t: None)
    assert lite[0].page_image.name == "page_1.png"
    engine = FakeEngine(["![text](0,0,1000,1000) 照片"])
    ocr = LightOnOcrParser(engine).parse(str(path), out, lambda d, t: None)
    assert ocr[0].text == "照片"
    r, g, b = engine.images[0].getpixel((32, 24))
    assert r > 150 and b < 80  # the first (red) frame, not the second


def test_16_bit_images_keep_their_contrast(tmp_path):
    from PIL import Image

    from app.kb.parsers.images import open_image, to_rgb

    path = tmp_path / "deep.png"
    Image.new("I;16", (4, 4), 300).save(path)
    image = open_image(path)
    assert image.mode == "RGB" and image.getpixel((0, 0)) != (255, 255, 255)
    assert to_rgb(Image.new("I;16", (2, 2), 300)).getpixel((0, 0)) == (1, 1, 1)
    # 32-bit integer images are scaled by their real range
    wide = Image.new("I", (2, 1), 0)
    wide.putpixel((1, 0), 70000)
    rgb = to_rgb(wide)
    assert (rgb.getpixel((0, 0)), rgb.getpixel((1, 0))) == ((0, 0, 0), (255, 255, 255))


def test_liteparse_lifts_its_page_cap_to_the_real_page_count(tmp_path, monkeypatch):
    import liteparse

    created = []
    real = liteparse.LiteParse

    def spy(**kwargs):
        created.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(liteparse, "LiteParse", spy)
    path = tmp_path / "a.pdf"
    path.write_bytes(make_pdf(["one", "two", "three"]))
    pages = LiteParseParser("eng").parse(str(path), tmp_path, lambda d, t: None)
    assert len(pages) == 3
    text_parsers = [k for k in created if k.get("output_format") == "markdown"]
    assert text_parsers and all(k["max_pages"] >= 3 for k in text_parsers)


def test_liteparse_ocrs_the_upright_page_image(tmp_path, monkeypatch):
    import liteparse

    parsed = []
    real = liteparse.LiteParse

    class Spy:
        def __init__(self, **kwargs):
            self._real = real(**kwargs)

        def parse(self, path):
            parsed.append(str(path))
            return self._real.parse(path)

    monkeypatch.setattr(liteparse, "LiteParse", Spy)
    path = _image_file(tmp_path / "photo.jpg", size=(300, 200), fmt="JPEG", orientation=6)
    out = tmp_path / "out"
    out.mkdir()
    LiteParseParser("eng").parse(str(path), out, lambda d, t: None)
    assert parsed == [str(out / "page_1.png")]


def test_lightonocr_caps_crops_so_the_images_field_never_drops_one(tmp_path):
    import json

    from app.kb.milvus_store import IMAGES_MAX_BYTES, _images_json
    from app.kb.parsers.base import ImageRef
    from app.kb.parsers.lightonocr import MAX_CROPS_PER_PAGE, LightOnOcrParser

    # the longest names and sizes ImageRef allows still fit when every crop is kept
    worst = [ImageRef("img_123456789_123456789.png", 999_999, 999_999)] * MAX_CROPS_PER_PAGE
    assert len(json.loads(_images_json(worst))) == MAX_CROPS_PER_PAGE
    assert len(_images_json(worst).encode()) <= IMAGES_MAX_BYTES

    from PIL import Image

    path = tmp_path / "many.png"
    Image.new("RGB", (1000, 1000), "white").save(path)
    raw = "\n".join(f"![image](0,0,10,10) 圖{n}" for n in range(MAX_CROPS_PER_PAGE + 5))
    out = tmp_path / "out"
    out.mkdir()
    chunk = LightOnOcrParser(FakeEngine([raw])).parse(str(path), out, lambda d, t: None)[0]
    assert len(chunk.images) == MAX_CROPS_PER_PAGE
    assert chunk.text.count("](img:") == MAX_CROPS_PER_PAGE
    # descriptions past the cap stay as text, without an image reference
    assert chunk.text.endswith(f"圖{MAX_CROPS_PER_PAGE + 4}")
    assert len(json.loads(_images_json(chunk.images))) == len(chunk.images)


# ---- LiteParse embedded images ----


def _save(path, size=(400, 300), color="red", fmt=None):
    from PIL import Image

    Image.new("RGB", size, color).save(path, format=fmt)
    return path


def test_liteparse_links_extracted_images(tmp_path):
    from PIL import Image

    from app.kb.parsers.base import ImageRef
    from app.kb.parsers.liteparse_parser import link_images

    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    _save(src / "img_p5_1.jpg", fmt="JPEG")
    _save(src / "img_p5_2.png", size=(3000, 1000), color="blue")
    markdown = "intro\n\n![](img_p5_1.jpg)\n\nmiddle\n\n![map](img_p5_2.png)\n\nend"
    text, refs = link_images(markdown, 5, src, out)
    assert text == (
        "intro\n\n![圖片 1](img:img_5_1.png)\n\nmiddle\n\n![圖片 2](img:img_5_2.png) map\n\nend"
    )
    assert refs == [ImageRef("img_5_1.png", 400, 300), ImageRef("img_5_2.png", 1536, 512)]
    with Image.open(out / "img_5_2.png") as image:
        assert (image.format, image.size) == ("PNG", (1536, 512))


def test_liteparse_drops_placeholders_without_a_usable_image(tmp_path):
    from app.kb.parsers.liteparse_parser import link_images

    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    (src / "img_p1_2.png").write_bytes(b"not an image")
    _save(src / "img_p1_3.png", size=(8, 8))  # a spacer, too small to mean anything
    _save(tmp_path / "outside.png")
    markdown = (
        "a ![](img_p1_1.png) b ![](img_p1_2.png) c ![](img_p1_3.png) "
        "d ![](../outside.png) e ![logo](img_p1_9.tif) f"
    )
    text, refs = link_images(markdown, 1, src, out)
    assert refs == []
    assert text == "a  b  c  d  e logo f"
    assert list(out.iterdir()) == []


def test_liteparse_caps_images_per_page(tmp_path, monkeypatch):
    from app.kb.parsers import liteparse_parser

    monkeypatch.setattr(liteparse_parser, "MAX_IMAGES_PER_PAGE", 2)
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    for i in (1, 2, 3):
        _save(src / f"img_p1_{i}.png")
    text, refs = liteparse_parser.link_images(
        "![](img_p1_1.png) ![](img_p1_2.png) ![x](img_p1_3.png)", 1, src, out
    )
    assert [r.name for r in refs] == ["img_1_1.png", "img_1_2.png"]
    assert text == "![圖片 1](img:img_1_1.png) ![圖片 2](img:img_1_2.png) x"


def _pdf_with_embedded_image(path):
    """A one-page PDF with a small picture placed on it (not a full-page scan)."""
    import pypdfium2 as pdfium
    from PIL import Image

    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(612, 792)
    image = pdfium.PdfImage.new(pdf)
    image.set_bitmap(pdfium.PdfBitmap.from_pil(Image.new("RGB", (400, 300), (200, 30, 30))))
    image.set_matrix(pdfium.PdfMatrix().scale(200, 150).translate(100, 400))
    page.insert_obj(image)
    page.gen_content()
    pdf.save(path)
    return path


def test_liteparse_appends_extracted_images_the_markdown_does_not_mention(tmp_path):
    from app.kb.parsers.liteparse_parser import link_images

    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    for name in ("img_p2_1.png", "img_p2_2.png", "img_p3_1.png"):
        _save(src / name)
    text, refs = link_images("text ![](img_p2_2.png)", 2, src, out)
    assert [r.name for r in refs] == ["img_2_1.png", "img_2_2.png"]
    # the linked image keeps its place; the unmentioned one of this page goes last
    assert text == "text ![圖片 1](img:img_2_1.png)\n\n![圖片 2](img:img_2_2.png)"


def test_liteparse_pdf_pages_carry_their_embedded_images(tmp_path):
    path = _pdf_with_embedded_image(tmp_path / "embed.pdf")
    out = tmp_path / "out"
    out.mkdir()
    (chunk,) = LiteParseParser("eng").parse(str(path), out, lambda d, t: None)
    assert chunk.kind == "page_image" and chunk.page_image.name == "page_1.png"
    assert [r.name for r in chunk.images] == ["img_1_1.png"]
    assert "![圖片 1](img:img_1_1.png)" in chunk.text
    assert "img_p" not in chunk.text
    assert sorted(p.name for p in out.iterdir()) == ["img_1_1.png", "page_1.png"]


# ---- thread safety of the shared native libraries ----


def test_pdfium_calls_hold_the_lock(tmp_path, monkeypatch):
    import pypdfium2

    from app.kb.parsers.liteparse_parser import pdf_page_count
    from app.kb.parsers.lightonocr import LightOnOcrParser
    from app.kb.parsers.pdfium_lock import PDFIUM_LOCK

    real_document = pypdfium2.PdfDocument

    def owned():
        assert PDFIUM_LOCK._is_owned(), "pdfium called without PDFIUM_LOCK"

    closed = {"pages": 0, "bitmaps": 0, "opened_pages": 0, "opened_bitmaps": 0}

    class Bitmap:
        def __init__(self, bitmap):
            self._bitmap = bitmap
            closed["opened_bitmaps"] += 1

        def close(self):
            owned()
            closed["bitmaps"] += 1
            self._bitmap.close()

        def to_pil(self):
            owned()
            return self._bitmap.to_pil()

    class Page:
        def __init__(self, page):
            self._page = page
            closed["opened_pages"] += 1

        def close(self):
            owned()
            closed["pages"] += 1
            self._page.close()

        def get_size(self):
            owned()
            return self._page.get_size()

        def render(self, *args, **kwargs):
            owned()
            return Bitmap(self._page.render(*args, **kwargs))

    class Document:
        def __init__(self, path):
            owned()
            self._pdf = real_document(path)

        def __len__(self):
            owned()
            return len(self._pdf)

        def __getitem__(self, index):
            owned()
            return Page(self._pdf[index])

        def close(self):
            owned()
            self._pdf.close()

    monkeypatch.setattr(pypdfium2, "PdfDocument", Document)
    path = tmp_path / "a.pdf"
    path.write_bytes(make_pdf(["one", "two", "three"]))
    assert pdf_page_count(str(path)) == 3
    pages = LightOnOcrParser(FakeEngine()).parse(str(path), tmp_path, lambda d, t: None)
    assert len(pages) == 3
    assert closed["opened_pages"] == closed["pages"] == 3
    assert closed["opened_bitmaps"] == closed["bitmaps"] == 3


def test_liteparse_parallel_matches_sequential(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "a.pdf"
    path.write_bytes(make_pdf([f"Page text number {n}" for n in range(1, 5)]))

    def run(name):
        out = tmp_path / name
        out.mkdir()
        pages = LiteParseParser("eng").parse(str(path), out, lambda d, t: None)
        for n in range(1, 5):
            assert (out / f"page_{n}.png").exists()
        return [(p.page, p.text) for p in pages]

    expected = run("sequential")
    assert len(expected) == 4
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(run, [f"parallel_{i}" for i in range(6)]))
    assert all(r == expected for r in results)


def test_transformers_engine_runs_one_ocr_at_a_time(monkeypatch):
    import contextlib
    import sys
    import threading
    import time
    import types
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from app.kb.parsers.ocr_engines import TransformersOcrEngine

    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(inference_mode=contextlib.nullcontext)
    )
    state = {"active": 0, "max": 0}
    guard = threading.Lock()

    class FakeInputs(dict):
        def to(self, device):
            return self

    class FakeProcessor:
        def apply_chat_template(self, conversation, **kwargs):
            return FakeInputs(input_ids=np.zeros((1, 3)))

        def decode(self, tokens, skip_special_tokens):
            return "x"

    class FakeModel:
        device = "cpu"

        def generate(self, **kwargs):
            with guard:
                state["active"] += 1
                state["max"] = max(state["max"], state["active"])
            time.sleep(0.05)
            with guard:
                state["active"] -= 1
            return np.zeros((1, 5))

    engine = TransformersOcrEngine("model", "http://ollama")
    engine._model, engine._processor = FakeModel(), FakeProcessor()
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda _: engine.ocr(object()), range(4)))
    assert state["max"] == 1


def test_liteparse_and_pdfium_run_together(tmp_path):
    """LiteParse bundles its own pdfium (separate library), so it needs no PDFIUM_LOCK:
    LiteParse parse+screenshot in two threads next to locked pypdfium2 renders in two others."""
    from concurrent.futures import ThreadPoolExecutor

    import pypdfium2

    from app.kb.parsers.pdfium_lock import PDFIUM_LOCK

    path = tmp_path / "big.pdf"
    path.write_bytes(make_pdf([f"Stress page {n}" for n in range(1, 21)]))

    def lite(i):
        out = tmp_path / f"lite_{i}"
        out.mkdir()
        pages = LiteParseParser("eng").parse(str(path), out, lambda d, t: None)
        assert all((out / f"page_{p.page}.png").exists() for p in pages)
        return [(p.page, p.text) for p in pages]

    def render(_):
        sizes = []
        for n in range(20):
            with PDFIUM_LOCK:
                pdf = pypdfium2.PdfDocument(str(path))
                try:
                    page = pdf[n]
                    try:
                        bitmap = page.render(scale=1)
                        try:
                            sizes.append(bitmap.to_pil().size)
                        finally:
                            bitmap.close()
                    finally:
                        page.close()
                finally:
                    pdf.close()
        return sizes

    expected = lite("seq")
    assert len(expected) == 20
    expected_sizes = render(0)
    with ThreadPoolExecutor(4) as pool:
        lites = [pool.submit(lite, i) for i in range(2)]
        renders = [pool.submit(render, i) for i in range(10)]
        assert all(f.result() == expected for f in lites)
        assert all(f.result() == expected_sizes for f in renders)


# ---- LightOnOCR: several pages in flight ----


def _fake_pdfium(monkeypatch, pages: int, rendered: list | None = None):
    """Replace pypdfium2.PdfDocument with a document whose page N renders to an image
    N + 100 pixels wide, so an engine can tell which page it received."""
    import pypdfium2
    from PIL import Image

    class Bitmap:
        def __init__(self, n):
            self.n = n

        def to_pil(self):
            if rendered is not None:
                rendered.append(self.n)
            return Image.new("RGB", (100 + self.n, 50), "white")

        def close(self):
            pass

    class Page:
        def __init__(self, n):
            self.n = n

        def get_size(self):
            return (100, 100)

        def render(self, **kwargs):
            return Bitmap(self.n)

        def close(self):
            pass

    class Document:
        def __init__(self, path):
            pass

        def __len__(self):
            return pages

        def __getitem__(self, i):
            return Page(i + 1)

        def close(self):
            pass

    monkeypatch.setattr(pypdfium2, "PdfDocument", Document)


class ConcurrentEngine:
    """Records which pages it OCRs and how many calls overlap."""

    def __init__(self, delay=lambda page: 0.05, fail_page=None):
        import threading

        self.delay = delay
        self.fail_page = fail_page
        self.lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.started: list[int] = []

    def availability(self):
        return True, ""

    def ocr(self, image):
        import time

        page = image.width - 100
        with self.lock:
            self.started.append(page)
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(self.delay(page))
            if page == self.fail_page:
                raise RuntimeError("boom")
            return f"![text](0,0,1000,1000) 第{page}頁\n![image](0,0,500,1000) 圖{page}"
        finally:
            with self.lock:
                self.in_flight -= 1

    def release(self):
        pass


def _pdf(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return str(path), out


def test_lightonocr_keeps_at_most_concurrency_pages_in_flight(tmp_path, monkeypatch):
    from app.kb.parsers.lightonocr import LightOnOcrParser

    _fake_pdfium(monkeypatch, 6)
    engine = ConcurrentEngine(delay=lambda page: 0.2)
    path, out = _pdf(tmp_path)
    pages = LightOnOcrParser(engine, concurrency=3).parse(path, out, lambda d, t: None)
    assert len(pages) == 6
    assert engine.max_in_flight == 3
    assert engine.in_flight == 0


def test_lightonocr_results_are_in_page_order(tmp_path, monkeypatch):
    import random

    from app.kb.parsers.lightonocr import LightOnOcrParser

    _fake_pdfium(monkeypatch, 6)
    rng = random.Random(7)
    delays = {p: rng.uniform(0, 0.1) for p in range(1, 7)}
    engine = ConcurrentEngine(delay=lambda page: delays[page])
    path, out = _pdf(tmp_path)
    pages = LightOnOcrParser(engine, concurrency=4).parse(path, out, lambda d, t: None)
    assert [p.page for p in pages] == [1, 2, 3, 4, 5, 6]
    for p in pages:
        assert p.text == f"第{p.page}頁\n\n![圖片 1](img:img_{p.page}_1.png) 圖{p.page}"
        assert [ref.name for ref in p.images] == [f"img_{p.page}_1.png"]
        assert (out / f"img_{p.page}_1.png").exists()


def test_lightonocr_progress_counts_every_page(tmp_path, monkeypatch):
    import threading

    from app.kb.parsers.lightonocr import LightOnOcrParser

    _fake_pdfium(monkeypatch, 6)
    engine = ConcurrentEngine(delay=lambda page: 0.1 if page % 2 else 0.01)
    path, out = _pdf(tmp_path)
    progress, threads = [], set()

    def on_progress(done, total):
        progress.append((done, total))
        threads.add(threading.get_ident())

    LightOnOcrParser(engine, concurrency=3).parse(path, out, on_progress)
    assert progress[0] == (0, 6) and progress[-1] == (6, 6)
    assert [d for d, _ in progress] == sorted(d for d, _ in progress)
    assert threads == {threading.get_ident()}  # only the parsing thread reports


def test_lightonocr_one_failed_page_fails_the_document(tmp_path, monkeypatch):
    from app.kb.parsers.base import ParseError
    from app.kb.parsers.lightonocr import LightOnOcrParser

    _fake_pdfium(monkeypatch, 8)
    # page 5 is still running when page 4's failure is seen: parse must wait for it
    engine = ConcurrentEngine(delay=lambda page: 0.3 if page == 5 else 0.01, fail_page=4)
    path, out = _pdf(tmp_path)
    with pytest.raises(ParseError, match="第 4 頁 OCR 失敗：boom"):
        LightOnOcrParser(engine, concurrency=2).parse(path, out, lambda d, t: None)
    assert engine.in_flight == 0  # nothing left running after parse raised
    # 1 and 2, then one new page per finished page: page 5 was the last one sent (it may
    # be cancelled before it starts), and nothing is sent once the failure is seen
    assert sorted(engine.started) in ([1, 2, 3, 4], [1, 2, 3, 4, 5])


def test_lightonocr_engine_parse_error_is_kept(tmp_path, monkeypatch):
    from app.kb.parsers.base import ParseError
    from app.kb.parsers.lightonocr import LightOnOcrParser

    class Engine(ConcurrentEngine):
        def ocr(self, image):
            raise ParseError("OCR 服務（vLLM）呼叫失敗：x")

    _fake_pdfium(monkeypatch, 3)
    path, out = _pdf(tmp_path)
    with pytest.raises(ParseError, match="^OCR 服務（vLLM）呼叫失敗：x$"):
        LightOnOcrParser(Engine(), concurrency=2).parse(path, out, lambda d, t: None)


def test_stop_while_pages_are_in_flight_indexes_nothing(tmp_path, monkeypatch):
    from app.kb.parsers.lightonocr import LightOnOcrParser

    class Stopped(Exception):
        pass

    stop = Stopped()

    def on_progress(done, total):
        if done == 2:
            raise stop

    _fake_pdfium(monkeypatch, 8)
    engine = ConcurrentEngine(delay=lambda page: 0.1)
    path, out = _pdf(tmp_path)
    with pytest.raises(Stopped) as raised:
        LightOnOcrParser(engine, concurrency=2).parse(path, out, on_progress)
    assert raised.value is stop
    assert engine.in_flight == 0
    # pages 1 and 2, plus page 3 sent after page 1 finished; nothing after the stop
    assert sorted(engine.started) in ([1, 2], [1, 2, 3])


def test_lightonocr_holds_few_rendered_pages(tmp_path, monkeypatch):
    from app.kb.parsers.lightonocr import LightOnOcrParser

    rendered: list[int] = []
    done = [0]
    alive: list[int] = []

    class Engine(ConcurrentEngine):
        def ocr(self, image):
            # pages rendered but not yet post-processed and reported
            alive.append(len(rendered) - done[0])
            return super().ocr(image)

    def on_progress(d, t):
        done[0] = d

    _fake_pdfium(monkeypatch, 10, rendered)
    path, out = _pdf(tmp_path)
    LightOnOcrParser(Engine(delay=lambda page: 0.02), concurrency=3).parse(
        path, out, on_progress
    )
    assert len(rendered) == 10
    assert max(alive) <= 3


class FingerprintEngine:
    """Answers by what the page image looks like, so the reply is page-specific even when
    several documents are parsed at once. Learns the replies in a sequential pass."""

    def __init__(self, delay=lambda: 0.0):
        import hashlib
        import threading

        self._hash = lambda image: hashlib.sha1(image.tobytes()).hexdigest()
        self.delay = delay
        self.replies: dict[str, str] = {}
        self.learning: str | None = None
        self.count = 0
        self.lock = threading.Lock()

    def availability(self):
        return True, ""

    def ocr(self, image):
        import time

        key = self._hash(image)
        with self.lock:
            if self.learning is not None:
                self.count += 1
                self.replies[key] = (
                    f"![text](0,0,1000,1000) {self.learning}頁{self.count}\n"
                    f"![image](0,0,500,1000) 圖{self.learning}{self.count}"
                )
            reply = self.replies[key]
        time.sleep(self.delay())
        return reply

    def release(self):
        pass


def test_lightonocr_two_documents_in_parallel_match_sequential(tmp_path):
    import random
    from concurrent.futures import ThreadPoolExecutor

    from app.kb.parsers.lightonocr import LightOnOcrParser

    rng = random.Random(3)
    engine = FingerprintEngine(delay=lambda: rng.uniform(0, 0.03))
    docs = {
        "a": make_pdf([f"alpha page {i}" for i in range(1, 5)]),
        "b": make_pdf([f"beta page {i}" for i in range(1, 4)]),
    }
    paths = {}
    for name, data in docs.items():
        paths[name] = tmp_path / f"{name}.pdf"
        paths[name].write_bytes(data)

    def snapshot(pages):
        return [(p.page, p.text, [ref.name for ref in p.images]) for p in pages]

    sequential = {}
    for name in docs:
        out = tmp_path / f"seq_{name}"
        out.mkdir()
        engine.learning, engine.count = name, 0
        parser = LightOnOcrParser(engine, concurrency=1)
        sequential[name] = snapshot(parser.parse(str(paths[name]), out, lambda d, t: None))
    engine.learning = None
    assert [len(sequential[n]) for n in docs] == [4, 3]

    parser = LightOnOcrParser(engine, concurrency=2)

    def run(name):
        out = tmp_path / f"par_{name}"
        out.mkdir()
        return snapshot(parser.parse(str(paths[name]), out, lambda d, t: None))

    with ThreadPoolExecutor(2) as pool:
        futures = {name: pool.submit(run, name) for name in docs}
        parallel = {name: f.result(timeout=60) for name, f in futures.items()}
    assert parallel == sequential

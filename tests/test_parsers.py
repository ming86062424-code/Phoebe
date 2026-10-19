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

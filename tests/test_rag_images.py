"""Which picture each image sent to the chat model is: the prompt names it."""

from io import BytesIO

from PIL import Image

from app.kb.index import Source
from app.kb.rag import SourceImage, build_system_prompt, load_source_images


def _png(path):
    buf = BytesIO()
    Image.new("RGB", (4, 3), "red").save(buf, format="PNG")
    path.write_bytes(buf.getvalue())


def _source(document_id, page, text, page_image=None, images=()):
    ref = lambda name: {"name": name, "width": 4, "height": 3}  # noqa: E731
    return Source(
        document_id=document_id,
        filename="a.pdf",
        page=page,
        score=0.5,
        text=text,
        kind="page_image" if page_image else "text",
        images=[ref(n) for n in images],
        page_image=ref(page_image) if page_image else None,
    )


def test_each_image_is_labelled_with_its_page_and_picture(tmp_path):
    for name in ("page_5.png", "img_5_1.png", "img_7_2.png", "img_7_9.png"):
        _png(tmp_path / name)
    sources = [
        _source(1, 5, "![圖片 1](img:img_5_1.png) 吉祥物", "page_5.png", ["img_5_1.png"]),
        _source(1, 7, "a ![圖片 2](img:img_7_2.png) map", None, ["img_7_2.png", "img_7_9.png"]),
    ]
    images = load_source_images(sources, lambda _doc: tmp_path, limit=4)
    assert [(i.source_number, i.label) for i in images] == [
        (1, "第 5 頁整頁"),
        (1, "第 5 頁的圖片 1"),
        (2, "第 7 頁的圖片 2"),
        (2, "第 7 頁的另一張圖"),  # not named in the text
    ]
    system = build_system_prompt("手冊", sources, images)
    for line in (
        "附圖 1 → [1] 第 5 頁整頁",
        "附圖 2 → [1] 第 5 頁的圖片 1",
        "附圖 3 → [2] 第 7 頁的圖片 2",
        "附圖 4 → [2] 第 7 頁的另一張圖",
    ):
        assert line in system


def test_text_file_images_say_full_text():
    system = build_system_prompt(
        "手冊",
        [_source(1, 0, "x")],
        [SourceImage(1, b"png", "全文的圖片 1")],
    )
    assert "附圖 1 → [1] 全文的圖片 1" in system

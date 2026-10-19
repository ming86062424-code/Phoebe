import base64
import io
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import create_app


class FakeModel:
    """Records every encode() call and returns deterministic, un-normalized vectors."""

    def __init__(self, dim=768):
        self.dim = dim
        self.calls = []

    def encode(self, inputs, **kwargs):
        self.calls.append((list(inputs), kwargs))
        rows = []
        for i, item in enumerate(inputs):
            if isinstance(item, str):
                key = item
            else:
                img = item["image"]
                key = (item.get("text"), img.size, img.getpixel((0, 0)))
            rng = np.random.default_rng(abs(hash(repr(key))) % (2**32))
            rows.append(rng.normal(size=self.dim) * 5 + 1)
        return np.asarray(rows, dtype=np.float32)


def png_b64(color="red", size=(8, 6)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def model():
    return FakeModel()


@pytest.fixture
def client(model):
    return TestClient(create_app(model=model))


def all_inputs(model):
    return [x for inputs, _ in model.calls for x in inputs]


def test_query_prefix(client, model):
    r = client.post("/embed", json={"items": [{"kind": "query", "text": "柴犬"}]})
    assert r.status_code == 200
    assert all_inputs(model) == ["task: search result | query: 柴犬"]


def test_document_prefix(client, model):
    client.post("/embed", json={"items": [{"kind": "document", "title": "doc.pdf", "text": "內容"}]})
    assert all_inputs(model) == ["title: doc.pdf | text: 內容"]


def test_image_has_no_prefix_and_uses_soft_tokens(client, model):
    r = client.post("/embed", json={"items": [{"kind": "image", "image_b64": png_b64()}]})
    assert r.status_code == 200
    (inputs, kwargs), = model.calls
    (item,) = inputs
    assert set(item) == {"image"}
    assert isinstance(item["image"], Image.Image)
    assert item["image"].size == (8, 6)
    assert kwargs["processing_kwargs"] == {"image": {"max_soft_tokens": 280}}


def test_image_tokens_configurable(model):
    c = TestClient(create_app(model=model, image_tokens=560))
    c.post("/embed", json={"items": [{"kind": "image", "image_b64": png_b64()}]})
    assert model.calls[0][1]["processing_kwargs"] == {"image": {"max_soft_tokens": 560}}


def test_image_text_uses_image_token(client, model):
    client.post(
        "/embed",
        json={"items": [{"kind": "image_text", "image_b64": png_b64(), "text": "第一頁"}]},
    )
    (inputs, kwargs), = model.calls
    (item,) = inputs
    assert item["text"] == "<|image|> 第一頁"
    assert isinstance(item["image"], Image.Image)
    assert kwargs["processing_kwargs"] == {"image": {"max_soft_tokens": 280}}


def test_output_is_l2_normalized_768(client):
    r = client.post(
        "/embed",
        json={
            "items": [
                {"kind": "query", "text": "a"},
                {"kind": "image", "image_b64": png_b64()},
            ]
        },
    )
    emb = np.asarray(r.json()["embeddings"])
    assert emb.shape == (2, 768)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


class KindModel:
    """One-hot vector per modality group (dims 0 query, 1 document, 2 image, 3 image_text);
    component 10 carries a per-item number so each row position is identifiable."""

    def encode(self, inputs, **kwargs):
        rows = []
        for item in inputs:
            v = np.zeros(768, dtype=np.float32)
            if isinstance(item, str):
                dim = 0 if item.startswith("task:") else 1
                n = int(item.rsplit("#", 1)[1])
            elif "text" in item:
                dim, n = 3, int(item["text"].rsplit("#", 1)[1])
            else:
                dim, n = 2, item["image"].size[0]
            v[dim] = 1.0
            v[10] = n
            rows.append(v)
        return np.asarray(rows)


def test_batch_keeps_input_order_across_kinds():
    c = TestClient(create_app(model=KindModel()))
    items = [
        {"kind": "query", "text": "#1"},
        {"kind": "image", "image_b64": png_b64(size=(11, 5))},
        {"kind": "document", "title": "t", "text": "#2"},
        {"kind": "image_text", "image_b64": png_b64(), "text": "#3"},
        {"kind": "query", "text": "#4"},
        {"kind": "image", "image_b64": png_b64(size=(15, 5))},
    ]
    emb = np.asarray(c.post("/embed", json={"items": items}).json()["embeddings"])
    dims = [int(np.argmax(r[:4])) for r in emb]
    assert dims == [0, 2, 1, 3, 0, 2]
    # normalization keeps the ratio between the one-hot component and the payload component
    ratio = emb[:, 10] / emb[np.arange(6), dims]
    assert np.allclose(ratio, [1, 11, 2, 3, 4, 15])


def test_empty_items_returns_empty(client, model):
    r = client.post("/embed", json={"items": []})
    assert r.status_code == 200
    assert r.json() == {"embeddings": []}
    assert model.calls == []


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"items": [{"kind": "nope", "text": "x"}]},
        {"items": [{"kind": "query"}]},
        {"items": [{"kind": "document", "text": "no title"}]},
        {"items": [{"kind": "image"}]},
        {"items": [{"kind": "image", "image_b64": "!!!not base64!!!"}]},
        {"items": [{"kind": "image", "image_b64": base64.b64encode(b"not an image").decode()}]},
        {"items": [{"kind": "image_text", "image_b64": png_b64()}]},
        {"items": [{"kind": "query", "text": "x"}] * 65},
    ],
)
def test_invalid_input_is_422(client, body):
    assert client.post("/embed", json=body).status_code == 422


def test_health_and_embed_503_until_model_loaded():
    gate = threading.Event()
    fake = FakeModel()

    def loader():
        gate.wait(5)
        return fake

    app = create_app(loader=loader)
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        r = c.post("/embed", json={"items": [{"kind": "query", "text": "x"}]})
        assert r.status_code == 503
        gate.set()
        app.state.loading_thread.join(5)
        assert c.get("/health").status_code == 200
        assert c.post("/embed", json={"items": [{"kind": "query", "text": "x"}]}).status_code == 200


def test_load_failure_keeps_503_and_reports_error():
    def loader():
        raise RuntimeError("boom")

    app = create_app(loader=loader, on_load_error=lambda: None)
    with TestClient(app) as c:
        app.state.loading_thread.join(5)
        r = c.get("/health")
        assert r.status_code == 503
        assert "boom" in r.json()["detail"]


def test_health_ok_with_injected_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def _b64_png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_oversized_base64_rejected_before_decoding(client):
    from app import MAX_IMAGE_B64_CHARS

    r = client.post(
        "/embed", json={"items": [{"kind": "image", "image_b64": "A" * (MAX_IMAGE_B64_CHARS + 4)}]}
    )
    assert r.status_code == 422


def test_too_many_pixels_rejected(client, monkeypatch):
    import app as app_module

    monkeypatch.setattr(app_module, "MAX_IMAGE_PIXELS", 100)
    r = client.post("/embed", json={"items": [{"kind": "image", "image_b64": png_b64(size=(11, 10))}]})
    assert r.status_code == 422
    ok = client.post("/embed", json={"items": [{"kind": "image", "image_b64": png_b64(size=(10, 10))}]})
    assert ok.status_code == 200


def test_decompression_bomb_warning_and_error_are_422(client, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 50)  # warns above 50 px, errors above 100 px
    for size in ((10, 6), (20, 10)):
        r = client.post("/embed", json={"items": [{"kind": "image", "image_b64": png_b64(size=size)}]})
        assert r.status_code == 422, size


def test_large_content_length_is_413(client):
    r = client.post(
        "/embed",
        content=b"{}",
        headers={"content-length": str(101 * 1024 * 1024), "content-type": "application/json"},
    )
    assert r.status_code == 413


def test_rgba_is_flattened_on_white(model):
    c = TestClient(create_app(model=model))
    b64 = _b64_png(Image.new("RGBA", (4, 4), (0, 0, 0, 0)))
    c.post("/embed", json={"items": [{"kind": "image", "image_b64": b64}]})
    img = model.calls[0][0][0]["image"]
    assert img.mode == "RGB" and img.getpixel((0, 0)) == (255, 255, 255)


def test_exif_orientation_applied(model):
    c = TestClient(create_app(model=model))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees: 8x4 becomes 4x8
    buf = io.BytesIO()
    Image.new("RGB", (8, 4), "red").save(buf, format="JPEG", exif=exif)
    b64 = base64.b64encode(buf.getvalue()).decode()
    c.post("/embed", json={"items": [{"kind": "image", "image_b64": b64}]})
    assert model.calls[0][0][0]["image"].size == (4, 8)


@pytest.mark.parametrize("bad", [0, 100, 281])
def test_invalid_image_tokens_rejected(bad):
    with pytest.raises(ValueError, match="EMBED_IMAGE_TOKENS"):
        create_app(model=FakeModel(), image_tokens=bad)


def test_load_failure_exits_process_by_default(monkeypatch):
    called = []
    monkeypatch.setattr("os._exit", lambda code: called.append(code))

    def loader():
        raise RuntimeError("boom")

    app = create_app(loader=loader)
    with TestClient(app):
        app.state.loading_thread.join(5)
    assert called == [1]

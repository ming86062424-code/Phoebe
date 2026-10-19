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


def test_batch_keeps_input_order_across_kinds(client, model):
    items = [
        {"kind": "query", "text": "q1"},
        {"kind": "image", "image_b64": png_b64("red")},
        {"kind": "document", "title": "t", "text": "d1"},
        {"kind": "image_text", "image_b64": png_b64("blue"), "text": "x"},
        {"kind": "query", "text": "q2"},
    ]
    emb = np.asarray(client.post("/embed", json={"items": items}).json()["embeddings"])
    assert emb.shape == (5, 768)
    # each item is embedded exactly once, and a repeated request gives the same order
    assert sum(len(i) for i, _ in model.calls) == 5
    again = np.asarray(client.post("/embed", json={"items": items}).json()["embeddings"])
    assert np.allclose(emb, again)
    # the two queries land in rows 0 and 4 (text group re-ordered back)
    single = np.asarray(
        client.post("/embed", json={"items": [{"kind": "query", "text": "q2"}]}).json()["embeddings"]
    )
    assert np.allclose(emb[4], single[0])


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

    app = create_app(loader=loader)
    with TestClient(app) as c:
        app.state.loading_thread.join(5)
        r = c.get("/health")
        assert r.status_code == 503
        assert "boom" in r.json()["detail"]


def test_health_ok_with_injected_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

import base64
import json
import os
import socket
from urllib.parse import urlparse

import httpx
import pytest

from app.kb.chunk_store import KnowledgeStoreError
from app.kb.embedder_client import BATCH_SIZE, EmbedderClient


def make_client(handler):
    return EmbedderClient("http://embedder:8000", transport=httpx.MockTransport(handler))


def echo(requests):
    def handler(request):
        body = json.loads(request.content)
        requests.append((request, body))
        return httpx.Response(
            200, json={"embeddings": [[float(i)] for i, _ in enumerate(body["items"])]}
        )

    return handler


def test_embed_queries_request_shape():
    seen = []
    out = make_client(echo(seen)).embed_queries(["a", "b"])
    request, body = seen[0]
    assert str(request.url) == "http://embedder:8000/embed"
    assert body == {"items": [{"kind": "query", "text": "a"}, {"kind": "query", "text": "b"}]}
    assert out == [[0.0], [1.0]]


def test_embed_documents_sends_title_and_text_separately():
    seen = []
    make_client(echo(seen)).embed_documents("規章.pdf", ["x", "y"])
    assert seen[0][1]["items"] == [
        {"kind": "document", "title": "規章.pdf", "text": "x"},
        {"kind": "document", "title": "規章.pdf", "text": "y"},
    ]


def test_embed_images_and_image_texts_use_base64():
    seen = []
    client = make_client(echo(seen))
    client.embed_images([b"\x89PNG"])
    client.embed_image_texts([(b"\x89PNG", "caption")])
    b64 = base64.b64encode(b"\x89PNG").decode()
    assert seen[0][1]["items"] == [{"kind": "image", "image_b64": b64}]
    assert seen[1][1]["items"] == [{"kind": "image_text", "image_b64": b64, "text": "caption"}]


def test_batches_are_at_most_16_and_order_is_kept():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(len(body["items"]))
        return httpx.Response(
            200, json={"embeddings": [[float(i["text"])] for i in body["items"]]}
        )

    texts = [str(i) for i in range(40)]
    out = make_client(handler).embed_queries(texts)
    assert BATCH_SIZE == 16
    assert seen == [16, 16, 8]
    assert out == [[float(i)] for i in range(40)]


def test_empty_input_makes_no_request():
    def handler(request):
        raise AssertionError("no request expected")

    assert make_client(handler).embed_queries([]) == []


def test_connection_error_is_translated():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(KnowledgeStoreError) as err:
        make_client(handler).embed_queries(["a"])
    assert str(err.value) == "無法連線到 embedding 服務"


def test_timeout_is_translated():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(KnowledgeStoreError, match="無法連線到 embedding 服務"):
        make_client(handler).embed_queries(["a"])


def test_model_loading_503():
    client = make_client(lambda r: httpx.Response(503, json={"detail": "model loading"}))
    with pytest.raises(KnowledgeStoreError) as err:
        client.embed_queries(["a"])
    assert str(err.value) == "無法連線到 embedding 服務（模型載入中）"


def test_other_http_error_is_reported():
    client = make_client(lambda r: httpx.Response(422, json={"detail": "bad"}))
    with pytest.raises(KnowledgeStoreError, match="embedding 服務回應錯誤 422"):
        client.embed_queries(["a"])


def test_wrong_count_is_an_error():
    client = make_client(lambda r: httpx.Response(200, json={"embeddings": []}))
    with pytest.raises(KnowledgeStoreError, match="embedding 服務"):
        client.embed_queries(["a"])


def test_default_timeout_is_120_seconds():
    assert EmbedderClient("http://x")._client.timeout == httpx.Timeout(120.0)


# ---- real embedder (skipped unless EMBEDDER_URL is reachable) ----


def _embedder_url() -> str:
    url = os.environ.get("EMBEDDER_URL", "http://localhost:8002")
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 80), timeout=1):
            pass
        if httpx.get(f"{url}/health", timeout=5).status_code != 200:
            raise OSError("not ready")
        return url
    except (OSError, httpx.HTTPError):
        pytest.skip(f"embedder not reachable at {url}")


@pytest.mark.embedder
def test_real_embedder_ranks_related_document_closer():
    client = EmbedderClient(_embedder_url())
    query = client.embed_queries(["日本的狗"])[0]
    dog, invoice = client.embed_documents(
        "doc.txt",
        ["柴犬是原產於日本的犬種，性格忠誠活潑。", "發票號碼 2026-1018，應付金額與付款期限。"],
    )
    assert len(query) == 768

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    assert dot(query, dog) > dot(query, invoice)

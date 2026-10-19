import json

import httpx
import pytest

from app.ollama_client import (
    ChatDelta,
    ModelNotFound,
    OllamaClient,
    OllamaStreamError,
    OllamaUnavailable,
)

pytestmark = pytest.mark.anyio

MODEL = "gemma4:e4b"


def ndjson(*objs) -> bytes:
    return b"".join(json.dumps(o, ensure_ascii=False).encode() + b"\n" for o in objs)


def make_client(handler, think: bool = True) -> OllamaClient:
    return OllamaClient(
        "http://ollama.test", MODEL, think=think, transport=httpx.MockTransport(handler)
    )


async def collect(client, messages=None):
    return [p async for p in client.stream_chat(messages or [{"role": "user", "content": "hi"}])]


class SplitStream(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


async def test_stream_chat_yields_content_pieces():
    body = ndjson(
        {"message": {"role": "assistant", "content": "你"}, "done": False},
        {"message": {"role": "assistant", "content": "好"}, "done": False},
        {"done": True},
    )
    client = make_client(lambda request: httpx.Response(200, content=body))
    assert await collect(client) == [ChatDelta("content", "你"), ChatDelta("content", "好")]


async def test_stream_chat_sends_model_messages_and_stream_flag():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=ndjson({"done": True}))

    messages = [{"role": "user", "content": "哈囉"}]
    await collect(make_client(handler), messages)
    assert seen["path"] == "/api/chat"
    assert seen["body"]["model"] == MODEL
    assert seen["body"]["stream"] is True
    assert seen["body"]["messages"] == messages


async def test_stream_chat_handles_multibyte_split_across_chunks():
    body = ndjson({"message": {"content": "柴犬"}}, {"done": True})
    split = body.index("犬".encode()) + 1
    stream = SplitStream(body[:split], body[split:])
    client = make_client(lambda request: httpx.Response(200, stream=stream))
    assert "".join([d.text for d in await collect(client)]) == "柴犬"


async def test_stream_chat_skips_empty_content():
    body = ndjson(
        {"message": {"content": ""}, "done": False},
        {"message": {"content": "好"}, "done": False},
        {"message": {"content": ""}, "done": True},
    )
    client = make_client(lambda request: httpx.Response(200, content=body))
    assert await collect(client) == [ChatDelta("content", "好")]


async def test_connect_error_raises_ollama_unavailable():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(OllamaUnavailable):
        await collect(make_client(handler))


async def test_connect_timeout_raises_ollama_unavailable():
    def handler(request):
        raise httpx.ConnectTimeout("slow")

    with pytest.raises(OllamaUnavailable):
        await collect(make_client(handler))


async def test_404_raises_model_not_found_with_model_name():
    body = json.dumps({"error": 'model "x" not found, try pulling it first'}).encode()
    client = make_client(lambda request: httpx.Response(404, content=body))
    with pytest.raises(ModelNotFound) as exc:
        await collect(client)
    assert MODEL in str(exc.value)


async def test_other_status_raises_stream_error():
    client = make_client(lambda request: httpx.Response(500, content=b"oops"))
    with pytest.raises(OllamaStreamError):
        await collect(client)


async def test_error_line_mid_stream_raises_stream_error():
    body = ndjson({"message": {"content": "你"}, "done": False}, {"error": "boom"})
    client = make_client(lambda request: httpx.Response(200, content=body))
    pieces = []
    with pytest.raises(OllamaStreamError, match="boom"):
        async for piece in client.stream_chat([{"role": "user", "content": "hi"}]):
            pieces.append(piece)
    assert pieces == [ChatDelta("content", "你")]


async def test_stream_chat_yields_thinking_then_content():
    body = ndjson(
        {"message": {"thinking": "想"}},
        {"message": {"content": "你"}},
        {"done": True},
    )
    client = make_client(lambda request: httpx.Response(200, content=body))
    assert await collect(client) == [ChatDelta("thinking", "想"), ChatDelta("content", "你")]


async def test_stream_chat_sends_think_flag():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content)["think"])
        return httpx.Response(200, content=ndjson({"done": True}))

    await collect(make_client(handler, think=False))
    await collect(make_client(handler))
    assert seen == [False, True]


async def test_same_line_thinking_before_content():
    body = ndjson({"message": {"thinking": "a", "content": "b"}}, {"done": True})
    client = make_client(lambda request: httpx.Response(200, content=body))
    assert await collect(client) == [ChatDelta("thinking", "a"), ChatDelta("content", "b")]


async def test_ollama_sends_system_and_temperature():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=ndjson({"done": True}))

    client = make_client(handler)
    messages = [{"role": "user", "content": "hi"}]
    async for _ in client.stream_chat(messages, system="你是助理", temperature=0.5):
        pass
    assert seen["body"]["messages"] == [{"role": "system", "content": "你是助理"}, *messages]
    assert seen["body"]["options"] == {"temperature": 0.5}
    assert messages == [{"role": "user", "content": "hi"}]


async def test_ollama_omits_temperature_when_none():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=ndjson({"done": True}))

    await collect(make_client(handler))
    assert "options" not in seen["body"]
    assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]


async def test_ollama_list_models_excludes_embedding_models():
    body = {"models": [{"name": "gemma4:e4b"}, {"name": "bge-m3:latest"}, {"name": "nomic-embed-text"}]}

    def handler(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json=body)

    assert await make_client(handler).list_models() == ["gemma4:e4b", "bge-m3:latest"]


async def test_ollama_availability():
    ok = make_client(lambda request: httpx.Response(200, json={"models": []}))
    assert await ok.availability() == (True, "")

    def refuse(request):
        raise httpx.ConnectError("refused")

    assert await make_client(refuse).availability() == (False, "無法連線到 Ollama")


# ---- images ----

PNG_A, PNG_B = b"\x89PNG-a", b"\x89PNG-b"


def _capture(seen, response=None):
    def handler(request):
        seen.append(json.loads(request.content))
        return response or httpx.Response(200, content=ndjson({"done": True}))

    return handler


async def test_images_are_sent_base64_on_the_last_user_message_only():
    import base64

    seen = []
    messages = [
        {"role": "user", "content": "一", "images": [PNG_B]},
        {"role": "assistant", "content": "二"},
        {"role": "user", "content": "三", "images": [PNG_A, PNG_B]},
    ]
    await collect(make_client(_capture(seen)), messages)
    sent = seen[0]["messages"]
    assert sent[-1] == {
        "role": "user",
        "content": "三",
        "images": [base64.b64encode(PNG_A).decode(), base64.b64encode(PNG_B).decode()],
    }
    assert sent[0] == {"role": "user", "content": "一"}
    assert sent[1] == {"role": "assistant", "content": "二"}
    # the caller's messages are not changed
    assert messages[-1]["images"] == [PNG_A, PNG_B]


async def test_messages_without_images_have_no_images_key():
    seen = []
    await collect(make_client(_capture(seen)), [{"role": "user", "content": "hi", "images": []}])
    assert seen[0]["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, json={"error": "this model is missing data required for image input"}),
        httpx.Response(400, json={"error": "llama3 does not support images"}),
        httpx.Response(400, json={"error": "model does not support vision"}),
        httpx.Response(200, content=ndjson({"error": "multimodal input is not supported"})),
    ],
)
async def test_rejected_images_raise_images_not_supported(response):
    from app.llm import ImagesNotSupported

    messages = [{"role": "user", "content": "hi", "images": [PNG_A]}]
    with pytest.raises(ImagesNotSupported):
        await collect(make_client(lambda request: response), messages)


async def test_image_like_errors_without_images_stay_ordinary_errors():
    from app.llm import ImagesNotSupported

    response = httpx.Response(400, json={"error": "model does not support images"})
    with pytest.raises(OllamaStreamError) as exc:
        await collect(make_client(lambda request: response))
    assert not isinstance(exc.value, ImagesNotSupported)
    # an unrelated failure while sending images is not mistaken for a rejection
    other = httpx.Response(500, json={"error": "out of memory"})
    with pytest.raises(OllamaStreamError) as exc:
        await collect(make_client(lambda request: other), [{"role": "user", "content": "x", "images": [PNG_A]}])
    assert not isinstance(exc.value, ImagesNotSupported)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("this model is missing data required for image input", True),
        ("model does not support vision", True),
        ("gemma3:1b does not support images", True),
        ("Image input modality is not enabled", True),
        ("llama3 does not support thinking", False),
        ("llama3 does not support tools", False),
        ("out of memory", False),
        ("", False),
    ],
)
def test_mentions_images(message, expected):
    from app.llm.base import mentions_images

    assert mentions_images(message) is expected


async def test_thinking_not_supported_is_not_an_image_rejection():
    from app.llm import ImagesNotSupported

    response = httpx.Response(400, json={"error": "llama3 does not support thinking"})
    with pytest.raises(OllamaStreamError) as exc:
        await collect(make_client(lambda request: response), [{"role": "user", "content": "x", "images": [PNG_A]}])
    assert not isinstance(exc.value, ImagesNotSupported)

import json

import httpx
import pytest

from app.ollama_client import (
    ModelNotFound,
    OllamaClient,
    OllamaStreamError,
    OllamaUnavailable,
)

pytestmark = pytest.mark.anyio

MODEL = "gemma4:e4b"


def ndjson(*objs) -> bytes:
    return b"".join(json.dumps(o, ensure_ascii=False).encode() + b"\n" for o in objs)


def make_client(handler) -> OllamaClient:
    return OllamaClient(
        "http://ollama.test", MODEL, transport=httpx.MockTransport(handler)
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
    assert await collect(client) == ["你", "好"]


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
    assert "".join(await collect(client)) == "柴犬"


async def test_stream_chat_skips_empty_content():
    body = ndjson(
        {"message": {"content": ""}, "done": False},
        {"message": {"content": "好"}, "done": False},
        {"message": {"content": ""}, "done": True},
    )
    client = make_client(lambda request: httpx.Response(200, content=body))
    assert await collect(client) == ["好"]


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
    assert pieces == ["你"]

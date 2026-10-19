import asyncio
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors, types

from app.config import Settings
from app.llm import (
    LlmAuthError,
    LlmRateLimited,
    LlmStreamError,
    LlmUnavailable,
    ModelNotFound,
    OllamaChatClient,
)
from app.llm.factory import build_chat_client
from app.llm import gemini as gemini_mod
from app.llm.gemini import GeminiChatClient
from app.llm_settings import LlmSettings

pytestmark = pytest.mark.anyio

MODEL = "gemini-x"
MSGS = [
    {"role": "user", "content": "a"},
    {"role": "assistant", "content": "b"},
    {"role": "user", "content": "c"},
]


def part(text, thought=None):
    return types.Part(text=text, thought=thought)


def resp(*parts):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))]
    )


def api_error(code, message="boom"):
    cls = errors.ServerError if code >= 500 else errors.ClientError
    return cls(code, {"error": {"code": code, "message": message, "status": "X"}})


class FakeModels:
    """Lazy like the SDK: errors surface on iteration, not on await."""

    def __init__(self, chunks=(), error=None, error_after=0, models=(), hang_after=None):
        self.chunks = list(chunks)
        self.error = error
        self.error_after = error_after
        self.models = list(models)
        self.hang_after = hang_after
        self.calls = []

    async def generate_content_stream(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})

        async def gen():
            for i, chunk in enumerate(self.chunks):
                if self.hang_after == i:
                    await asyncio.Event().wait()
                if self.error is not None and i == self.error_after:
                    raise self.error
                yield chunk
            if self.hang_after is not None and self.hang_after >= len(self.chunks):
                await asyncio.Event().wait()
            if self.error is not None and self.error_after >= len(self.chunks):
                raise self.error

        return gen()

    async def list(self, *, config=None):
        self.list_config = config
        if self.error is not None:
            raise self.error

        async def gen():
            for i, m in enumerate(self.models):
                if self.hang_after == i:
                    await asyncio.Event().wait()
                yield m

        return gen()


def make(models, think=True):
    fake = SimpleNamespace(aio=SimpleNamespace(models=models))
    return GeminiChatClient("k", MODEL, think=think, client=fake)


async def collect(client, **kw):
    return [d async for d in client.stream_chat(MSGS, **kw)]


async def test_gemini_maps_roles_and_config():
    models = FakeModels([resp(part("x"))])
    await collect(make(models, think=False), system="sys", temperature=0.5)
    call = models.calls[0]
    assert call["model"] == MODEL
    assert [c.role for c in call["contents"]] == ["user", "model", "user"]
    assert call["contents"][1].parts[0].text == "b"
    assert call["config"].system_instruction == "sys"
    assert call["config"].temperature == 0.5
    assert call["config"].thinking_config.include_thoughts is False


async def test_gemini_yields_thought_and_content():
    models = FakeModels(
        [resp(part("想", True)), resp(part("")), resp(part("你"), part("好", False))]
    )
    out = await collect(make(models))
    assert [(d.kind, d.text) for d in out] == [
        ("thinking", "想"),
        ("content", "你"),
        ("content", "好"),
    ]


@pytest.mark.parametrize("code", [401, 403])
async def test_gemini_401_403_maps_to_auth_error(code):
    with pytest.raises(LlmAuthError, match="Gemini API key 無效"):
        await collect(make(FakeModels(error=api_error(code))))


async def test_gemini_400_api_key_vs_other():
    with pytest.raises(LlmAuthError):
        await collect(make(FakeModels(error=api_error(400, "API key not valid"))))
    with pytest.raises(LlmStreamError) as ei:
        await collect(make(FakeModels(error=api_error(400, "bad request"))))
    assert not isinstance(ei.value, LlmAuthError)


async def test_gemini_404_maps_to_model_not_found():
    with pytest.raises(ModelNotFound, match="找不到模型 gemini-x"):
        await collect(make(FakeModels(error=api_error(404))))


async def test_gemini_429_maps_to_rate_limited():
    with pytest.raises(LlmRateLimited, match="Gemini 配額已用完或請求太頻繁，請稍後再試"):
        await collect(make(FakeModels(error=api_error(429))))


async def test_gemini_mid_stream_error_keeps_class():
    models = FakeModels([resp(part("a"))], error=api_error(429), error_after=1)
    got = []
    with pytest.raises(LlmRateLimited):
        async for d in make(models).stream_chat(MSGS):
            got.append(d.text)
    assert got == ["a"]


@pytest.mark.parametrize(
    "err",
    [
        api_error(503),
        httpx.ConnectError("x"),
        httpx.ReadTimeout("x"),
        TimeoutError(),
        ConnectionError(),
    ],
)
async def test_gemini_5xx_and_connect_map_to_unavailable(err):
    with pytest.raises(LlmUnavailable, match="無法連線到 Gemini"):
        await collect(make(FakeModels(error=err)))


async def test_gemini_error_message_has_no_key():
    fake = SimpleNamespace(
        aio=SimpleNamespace(models=FakeModels(error=api_error(403, "bad sk-secret")))
    )
    client = GeminiChatClient("sk-secret", MODEL, think=True, client=fake)
    with pytest.raises(LlmAuthError) as ei:
        await collect(client)
    assert "sk-secret" not in str(ei.value)


async def test_gemini_list_models_filters_and_sorts():
    def m(name, actions):
        return types.Model(name=name, supported_actions=actions)

    models = FakeModels(
        models=[
            m("models/gemini-2.5-pro", ["generateContent"]),
            m("models/gemini-2.5-flash", ["generateContent", "countTokens"]),
            m("models/gemini-embedding-001", ["embedContent"]),
            m("models/gemma-3-27b", ["generateContent"]),
            m("models/gemini-none", None),
        ]
    )
    assert await make(models).list_models() == ["gemini-2.5-flash", "gemini-2.5-pro"]


async def test_gemini_first_chunk_timeout(monkeypatch):
    monkeypatch.setattr(gemini_mod, "OPEN_TIMEOUT", 0.05)
    with pytest.raises(LlmUnavailable, match="Gemini 回應逾時"):
        await collect(make(FakeModels(hang_after=0, chunks=[resp(part("x"))])))


async def test_gemini_unparsable_response_never_escapes():
    err = errors.UnknownApiResponseError("junk")
    with pytest.raises(LlmUnavailable):
        await collect(make(FakeModels(chunks=[resp(part("a"))], error=err)))
    models = FakeModels(chunks=[resp(part("a"))], error=err, error_after=1)
    with pytest.raises(LlmStreamError):
        await collect(make(models))


async def test_gemini_list_models_whole_pagination_timeout(monkeypatch):
    monkeypatch.setattr(gemini_mod, "LIST_TIMEOUT", 0.05)
    m = types.Model(name="models/gemini-a", supported_actions=["generateContent"])
    with pytest.raises(LlmUnavailable, match="無法連線到 Gemini"):
        await make(FakeModels(models=[m, m], hang_after=1)).list_models()


async def test_gemini_list_models_error_maps():
    with pytest.raises(LlmAuthError):
        await make(FakeModels(error=api_error(403))).list_models()


async def test_gemini_availability_without_key():
    c = GeminiChatClient("", MODEL, think=True, client=object())
    assert await c.availability() == (False, "未設定 GEMINI_API_KEY")
    assert await make(FakeModels()).availability() == (True, "")


def test_factory_builds_client_for_provider():
    s = Settings(gemini_api_key="k")
    base = dict(system_prompt="", temperature=None, ollama_model="o", gemini_model="g")
    c = build_chat_client(s, LlmSettings(llm_provider="gemini", **base))
    assert isinstance(c, GeminiChatClient) and c.model == "g"
    c = build_chat_client(s, LlmSettings(llm_provider="ollama", **base))
    assert isinstance(c, OllamaChatClient) and c.model == "o"


async def test_gemini_skips_empty_history_messages():
    models = FakeModels([resp(part("x"))])
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "   "},
        {"role": "user", "content": "c"},
    ]
    [d async for d in make(models).stream_chat(msgs)]
    texts = [c.parts[0].text for c in models.calls[0]["contents"]]
    assert texts == ["a", "c"]


def blocked_resp(finish=None, block=None):
    return types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=block)
        if block
        else None,
        candidates=[types.Candidate(finish_reason=finish)] if finish else None,
    )


@pytest.mark.parametrize(
    "chunk",
    [
        blocked_resp(block=types.BlockedReason.SAFETY),
        blocked_resp(finish=types.FinishReason.SAFETY),
        blocked_resp(finish=types.FinishReason.PROHIBITED_CONTENT),
        blocked_resp(finish=types.FinishReason.RECITATION),
    ],
)
async def test_gemini_blocked_reply_raises(chunk):
    with pytest.raises(LlmStreamError, match="Gemini 拒絕回覆（內容安全限制）"):
        await collect(make(FakeModels([chunk])))


async def test_gemini_stop_without_text_is_not_blocked():
    out = await collect(make(FakeModels([blocked_resp(finish=types.FinishReason.STOP)])))
    assert out == []


async def test_gemini_text_then_safety_stop_is_not_error():
    models = FakeModels([resp(part("hi")), blocked_resp(finish=types.FinishReason.SAFETY)])
    assert [d.text for d in await collect(make(models))] == ["hi"]


def test_sdk_client_is_cached_per_api_key(monkeypatch):
    created = []
    monkeypatch.setattr(gemini_mod.genai, "Client", lambda **kw: created.append(kw) or object())
    gemini_mod._CLIENTS.clear()
    a = GeminiChatClient("k1", "m", True)._sdk()
    b = GeminiChatClient("k1", "m2", False)._sdk()
    c = GeminiChatClient("k2", "m", True)._sdk()
    assert a is b and a is not c and len(created) == 2
    gemini_mod._CLIENTS.clear()


async def test_gemini_closes_inner_iterator_on_early_exit():
    closed = []

    class Models(FakeModels):
        async def generate_content_stream(self, **kw):
            async def gen():
                try:
                    yield resp(part("a"))
                    yield resp(part("b"))
                finally:
                    closed.append(True)

            return gen()

    agen = make(Models()).stream_chat(MSGS)
    await agen.__anext__()
    await agen.aclose()
    assert closed == [True]


@pytest.mark.parametrize("error_after", [0, 1])
async def test_gemini_unexpected_exception_becomes_stream_error(error_after):
    models = FakeModels([resp(part("a"))], error=ValueError("secret sk-x"), error_after=error_after)
    with pytest.raises(LlmStreamError, match="^Gemini 回應錯誤$"):
        await collect(make(models))


async def test_gemini_cancel_is_not_swallowed():
    models = FakeModels(hang_after=0, chunks=[resp(part("x"))])
    task = asyncio.ensure_future(collect(make(models)))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_gemini_list_models_excludes_non_chat_variants():
    def m(name):
        return types.Model(name=f"models/{name}", supported_actions=["generateContent"])

    names = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-preview-tts",
        "gemini-2.0-flash-image",
        "gemini-live-2.5-flash",
        "gemini-embedding-exp",
        "gemini-2.5-flash-native-audio",
    ]
    out = await make(FakeModels(models=[m(n) for n in names])).list_models()
    assert out == ["gemini-2.5-flash"]

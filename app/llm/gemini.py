import asyncio
from collections.abc import AsyncIterator

import httpx
from google import genai
from google.genai import errors, types

from app.llm.base import (
    ChatDelta,
    LlmAuthError,
    LlmRateLimited,
    LlmStreamError,
    LlmUnavailable,
    ModelNotFound,
)

OPEN_TIMEOUT = 30.0
UNAVAILABLE = "無法連線到 Gemini"
AUTH_FAILED = "Gemini API key 無效"
RATE_LIMITED = "Gemini 配額已用完或請求太頻繁，請稍後再試"

# Transport-level failures: httpx, asyncio timeouts, sockets (and aiohttp, which
# the SDK uses when installed; its errors subclass OSError/TimeoutError or
# aiohttp.ClientError).
_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    httpx.HTTPError,
    asyncio.TimeoutError,
    OSError,
)
try:  # pragma: no cover - depends on the environment
    import aiohttp

    _TRANSPORT_ERRORS += (aiohttp.ClientError,)
except ImportError:  # pragma: no cover
    pass


class GeminiChatClient:
    def __init__(self, api_key: str, model: str, think: bool = True, client=None):
        self.api_key = api_key
        self.model = model
        self.think = think
        self._client = client

    def _sdk(self):
        if self._client is None:
            # No HttpOptions.timeout: it is a total deadline, which would cut
            # long streams. Opening the request is bounded by OPEN_TIMEOUT below.
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _map(self, e: BaseException) -> Exception:
        if isinstance(e, errors.APIError):
            code = e.code
            if code in (401, 403):
                return LlmAuthError(AUTH_FAILED)
            if code == 400:
                if "api key" in (e.message or "").lower():
                    return LlmAuthError(AUTH_FAILED)
                return LlmStreamError(e.message or f"Gemini 回應錯誤 {code}")
            if code == 404:
                return ModelNotFound(self.model)
            if code == 429:
                return LlmRateLimited(RATE_LIMITED)
            if code >= 500:
                return LlmUnavailable(UNAVAILABLE)
            return LlmStreamError(e.message or f"Gemini 回應錯誤 {code}")
        return LlmUnavailable(UNAVAILABLE)

    async def stream_chat(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[ChatDelta]:
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
        ]
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            thinking_config=types.ThinkingConfig(include_thoughts=self.think),
        )
        started = False
        try:
            # The SDK builds the generator on await; the HTTP request happens on
            # the first __anext__, so the open timeout bounds the first chunk.
            stream = await self._sdk().aio.models.generate_content_stream(
                model=self.model, contents=contents, config=config
            )
            it = stream.__aiter__()
            try:
                chunk = await asyncio.wait_for(anext(it), OPEN_TIMEOUT)
            except StopAsyncIteration:
                return
            while True:
                started = True
                for candidate in chunk.candidates or []:
                    content = candidate.content
                    for p in (content.parts if content else None) or []:
                        if not p.text:
                            continue
                        yield ChatDelta("thinking" if p.thought else "content", p.text)
                try:
                    chunk = await anext(it)
                except StopAsyncIteration:
                    return
        except errors.UnknownApiResponseError:
            if started:
                raise LlmStreamError("Gemini 回傳無法解析的資料") from None
            raise LlmUnavailable(UNAVAILABLE) from None
        except (errors.APIError, *_TRANSPORT_ERRORS) as e:
            raise self._map(e) from None

    async def list_models(self) -> list[str]:
        async def collect() -> list[str]:
            names: list[str] = []
            pager = await self._sdk().aio.models.list(config={"page_size": 1000})
            async for m in pager:
                name = (m.name or "").removeprefix("models/")
                if "generateContent" in (m.supported_actions or []) and name.startswith(
                    "gemini"
                ):
                    names.append(name)
            return names

        try:
            names = await asyncio.wait_for(collect(), OPEN_TIMEOUT)
        except errors.UnknownApiResponseError:
            raise LlmStreamError("Gemini 回傳無法解析的資料") from None
        except (errors.APIError, *_TRANSPORT_ERRORS) as e:
            raise self._map(e) from None
        return sorted(names)

    async def availability(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "未設定 GEMINI_API_KEY"
        return True, ""

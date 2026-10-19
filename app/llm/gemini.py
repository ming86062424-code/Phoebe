import asyncio
import logging
import threading
from collections.abc import AsyncIterator

import httpx
from google import genai
from google.genai import errors, types

from app.llm.base import (
    ChatDelta,
    ImagesNotSupported,
    LlmAuthError,
    LlmError,
    LlmRateLimited,
    LlmStreamError,
    LlmUnavailable,
    ModelNotFound,
    last_user_index,
    mentions_images,
)

logger = logging.getLogger(__name__)

OPEN_TIMEOUT = 120.0  # first chunk of a chat stream
LIST_TIMEOUT = 30.0  # whole model listing
TIMED_OUT = "Gemini 回應逾時"
BLOCKED = "Gemini 拒絕回覆（內容安全限制）"
NON_CHAT_MARKERS = ("tts", "image", "live", "embedding", "audio")
BLOCKING_FINISH_REASONS = {
    types.FinishReason.SAFETY,
    types.FinishReason.PROHIBITED_CONTENT,
    types.FinishReason.BLOCKLIST,
    types.FinishReason.SPII,
    types.FinishReason.RECITATION,
}
_CLIENTS: dict[str, object] = {}
_CLIENTS_LOCK = threading.Lock()


def _shared_client(api_key: str):
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(api_key)
        if client is None:
            # No HttpOptions.timeout: it is a total deadline, which would cut
            # long streams. Opening the request is bounded by OPEN_TIMEOUT.
            client = _CLIENTS[api_key] = genai.Client(api_key=api_key)
        return client


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
            self._client = _shared_client(self.api_key)
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
        last = last_user_index(messages)
        contents = []
        with_images = False
        for i, m in enumerate(messages):
            images = (m.get("images") or []) if i == last else []
            parts = [types.Part.from_bytes(data=b, mime_type="image/png") for b in images]
            if m["content"].strip():
                parts.append(types.Part(text=m["content"]))
            if parts:
                with_images = with_images or bool(images)
                contents.append(
                    types.Content(
                        role="model" if m["role"] == "assistant" else "user", parts=parts
                    )
                )
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            thinking_config=types.ThinkingConfig(include_thoughts=self.think),
        )
        started = False
        emitted = False
        blocked = False
        it = None
        try:
            # The SDK builds the generator on await; the HTTP request happens on
            # the first __anext__, so the open timeout bounds the first chunk.
            stream = await self._sdk().aio.models.generate_content_stream(
                model=self.model, contents=contents, config=config
            )
            it = stream.__aiter__()
            try:
                async with asyncio.timeout(OPEN_TIMEOUT) as deadline:
                    chunk = await anext(it)
            except StopAsyncIteration:
                return
            except TimeoutError:
                # only our own deadline; SDK/transport timeouts stay "無法連線"
                if deadline.expired():
                    raise LlmUnavailable(TIMED_OUT) from None
                raise
            while True:
                started = True
                feedback = getattr(chunk, "prompt_feedback", None)
                if feedback is not None and feedback.block_reason:
                    blocked = True
                for candidate in chunk.candidates or []:
                    if candidate.finish_reason in BLOCKING_FINISH_REASONS:
                        blocked = True
                    content = candidate.content
                    for p in (content.parts if content else None) or []:
                        if not p.text:
                            continue
                        emitted = True
                        yield ChatDelta("thinking" if p.thought else "content", p.text)
                try:
                    chunk = await anext(it)
                except StopAsyncIteration:
                    break
            if blocked and not emitted:
                raise LlmStreamError(BLOCKED)
        except LlmError:
            raise
        except errors.UnknownApiResponseError:
            if started:
                raise LlmStreamError("Gemini 回傳無法解析的資料") from None
            raise LlmUnavailable(UNAVAILABLE) from None
        except (errors.APIError, *_TRANSPORT_ERRORS) as e:
            if (
                with_images
                and isinstance(e, errors.APIError)
                and e.code == 400
                and mentions_images(e.message)
            ):
                raise ImagesNotSupported(e.message or "Gemini 無法處理圖片") from None
            raise self._map(e) from None
        except Exception as e:
            logger.warning("Gemini unexpected error: %s", type(e).__name__)
            raise LlmStreamError("Gemini 回應錯誤") from None
        finally:
            aclose = getattr(it, "aclose", None)
            if aclose is not None:
                await aclose()

    async def list_models(self) -> list[str]:
        async def collect() -> list[str]:
            names: list[str] = []
            pager = await self._sdk().aio.models.list(config={"page_size": 1000})
            async for m in pager:
                name = (m.name or "").removeprefix("models/")
                if (
                    "generateContent" in (m.supported_actions or [])
                    and name.startswith("gemini")
                    and not any(x in name for x in NON_CHAT_MARKERS)
                ):
                    names.append(name)
            return names

        try:
            names = await asyncio.wait_for(collect(), LIST_TIMEOUT)
        except LlmError:
            raise
        except errors.UnknownApiResponseError:
            raise LlmStreamError("Gemini 回傳無法解析的資料") from None
        except (errors.APIError, *_TRANSPORT_ERRORS) as e:
            raise self._map(e) from None
        except Exception as e:
            logger.warning("Gemini unexpected error: %s", type(e).__name__)
            raise LlmStreamError("Gemini 回應錯誤") from None
        return sorted(names)

    async def availability(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "未設定 GEMINI_API_KEY"
        return True, ""

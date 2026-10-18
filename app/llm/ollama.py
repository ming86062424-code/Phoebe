import json
from collections.abc import AsyncIterator

import httpx

from app.llm.base import ChatDelta, LlmStreamError, LlmUnavailable, ModelNotFound


class OllamaChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        think: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.think = think
        self._transport = transport

    async def stream_chat(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[ChatDelta]:
        if system:
            messages = [{"role": "system", "content": system}, *messages]
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": self.think,
        }
        if temperature is not None:
            payload["options"] = {"temperature": temperature}
        timeout = httpx.Timeout(None, connect=5.0)
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=timeout
            ) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/api/chat", json=payload
                ) as response:
                    if response.status_code != 200:
                        body = (await response.aread()).decode("utf-8", "replace")
                        if response.status_code == 404:
                            raise ModelNotFound(self.model)
                        raise LlmStreamError(
                            f"Ollama 回應錯誤 {response.status_code}: {body}"
                        )
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError as e:
                            raise LlmStreamError("Ollama 回傳無法解析的資料") from e
                        if "error" in data:
                            raise LlmStreamError(str(data["error"]))
                        message = data.get("message") or {}
                        if message.get("thinking"):
                            yield ChatDelta("thinking", message["thinking"])
                        if message.get("content"):
                            yield ChatDelta("content", message["content"])
                        if data.get("done"):
                            return
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise LlmUnavailable() from e

    async def _tags(self, timeout: httpx.Timeout | None = None) -> list[str]:
        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                timeout=timeout or httpx.Timeout(10.0, connect=5.0),
            ) as client:
                response = await client.get(f"{self.base_url}/api/tags")
        except httpx.HTTPError as e:
            raise LlmUnavailable() from e
        if response.status_code != 200:
            raise LlmStreamError(f"Ollama 回應錯誤 {response.status_code}")
        try:
            return [m["name"] for m in response.json().get("models") or []]
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise LlmStreamError("Ollama 回傳無法解析的資料") from e

    async def list_models(self) -> list[str]:
        return [name for name in await self._tags() if "embed" not in name.lower()]

    async def availability(self) -> tuple[bool, str]:
        try:
            await self._tags(httpx.Timeout(3.0))
        except Exception as e:
            return False, str(e)
        return True, ""

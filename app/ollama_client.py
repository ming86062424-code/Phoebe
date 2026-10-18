import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

import httpx


@dataclass(frozen=True)
class ChatDelta:
    kind: Literal["thinking", "content"]
    text: str


class OllamaError(Exception):
    pass


class OllamaUnavailable(OllamaError):
    def __init__(self, message: str = "無法連線到 Ollama"):
        super().__init__(message)


class ModelNotFound(OllamaError):
    def __init__(self, model: str):
        super().__init__(f"找不到模型 {model}")
        self.model = model


class OllamaStreamError(OllamaError):
    pass


class OllamaClient:
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

    async def stream_chat(self, messages: list[dict]) -> AsyncIterator[ChatDelta]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": self.think,
        }
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
                        raise OllamaStreamError(
                            f"Ollama 回應錯誤 {response.status_code}: {body}"
                        )
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError as e:
                            raise OllamaStreamError("Ollama 回傳無法解析的資料") from e
                        if "error" in data:
                            raise OllamaStreamError(str(data["error"]))
                        message = data.get("message") or {}
                        if message.get("thinking"):
                            yield ChatDelta("thinking", message["thinking"])
                        if message.get("content"):
                            yield ChatDelta("content", message["content"])
                        if data.get("done"):
                            return
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise OllamaUnavailable() from e

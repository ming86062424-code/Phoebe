from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class ChatDelta:
    kind: Literal["thinking", "content"]
    text: str


class LlmError(Exception):
    pass


class LlmUnavailable(LlmError):
    def __init__(self, message: str = "無法連線到 Ollama"):
        super().__init__(message)


class ModelNotFound(LlmError):
    def __init__(self, model: str):
        super().__init__(f"找不到模型 {model}")
        self.model = model


class LlmStreamError(LlmError):
    pass


class LlmAuthError(LlmError):
    pass


class LlmRateLimited(LlmError):
    pass


class ChatClient(Protocol):
    def stream_chat(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[ChatDelta]: ...

    async def list_models(self) -> list[str]: ...

    async def availability(self) -> tuple[bool, str]: ...

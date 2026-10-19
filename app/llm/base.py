import re
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


class ImagesNotSupported(LlmStreamError):
    """The model refused a request because it carried images (e.g. a text-only model)."""


# How providers phrase "this model can't take images". Only these phrasings count, so an
# unrelated error that merely names a vision model (llama3.2-vision) is never retried.
_IMAGE_REJECTION = re.compile(
    r"does not support (?:the )?(?:images?|vision|multimodal)"
    r"|(?:images?|vision|multimodal)(?: input)? (?:is |are )?not supported"
    r"|missing data required for image input"
    r"|image input modality is not enabled"
    r"|unable to process input image"
)


def mentions_images(message: str | None) -> bool:
    """Whether a provider's error message (for a request that carried images) says the
    model can't take images. Pass only the error message, not a whole response body."""
    return bool(_IMAGE_REJECTION.search((message or "").lower()))


def last_user_index(messages: list[dict]) -> int | None:
    """Index of the last user message: the only one whose `images` are sent."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return None


class ChatClient(Protocol):
    """messages are {"role", "content"} dicts; the last user message may also carry
    `images: list[bytes]` (PNG). A request rejected because of its images raises
    ImagesNotSupported before anything is yielded."""

    def stream_chat(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[ChatDelta]: ...

    async def list_models(self) -> list[str]: ...

    async def availability(self) -> tuple[bool, str]: ...

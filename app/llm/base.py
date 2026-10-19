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


# Words in a provider's error that point at the attached images.
IMAGE_REJECTION_MARKERS = ("image", "vision", "multimodal", "modality")
# Ollama also says "<model> does not support thinking" / "... tools": not about images.
OTHER_CAPABILITIES = ("think", "tool")


def mentions_images(message: str | None) -> bool:
    """Whether a provider error (for a request that carried images) blames the images."""
    text = (message or "").lower()
    if any(marker in text for marker in IMAGE_REJECTION_MARKERS):
        return True
    return "does not support" in text and not any(w in text for w in OTHER_CAPABILITIES)


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

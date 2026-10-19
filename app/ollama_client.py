"""Backward-compatible names; the implementation lives in app.llm."""

from app.llm import (
    ChatDelta,
    LlmError as OllamaError,
    LlmStreamError as OllamaStreamError,
    LlmUnavailable as OllamaUnavailable,
    ModelNotFound,
    OllamaChatClient as OllamaClient,
)

__all__ = [
    "ChatDelta",
    "ModelNotFound",
    "OllamaClient",
    "OllamaError",
    "OllamaStreamError",
    "OllamaUnavailable",
]

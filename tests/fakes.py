from app.ollama_client import ChatDelta


class FakeChatClient:
    def __init__(
        self,
        deltas: list[ChatDelta] | None = None,
        error: Exception | None = None,
        error_after: int | None = None,
    ):
        self.deltas = (
            [ChatDelta("thinking", "嗯"), ChatDelta("content", "你"), ChatDelta("content", "好")]
            if deltas is None
            else deltas
        )
        self.error = error
        self.error_after = error_after
        self.calls: list[list[dict]] = []

    async def stream_chat(self, messages: list[dict]):
        self.calls.append([dict(m) for m in messages])
        if self.error is not None and self.error_after is None:
            raise self.error
        for i, delta in enumerate(self.deltas):
            if self.error is not None and i == self.error_after:
                raise self.error
            yield delta
        if self.error is not None and self.error_after is not None and self.error_after >= len(self.deltas):
            raise self.error


import hashlib
import re

from llama_index.core.base.embeddings.base import BaseEmbedding


class KeywordEmbedding(BaseEmbedding):
    """Deterministic bag-of-words embedding so retrieval order is predictable in tests."""

    dim: int = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        if not any(v):
            v[0] = 1.0
        return v

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vec(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vec(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vec(query)


class FakeParser:
    id = "fake"
    label = "假解析器"

    def __init__(self, pages=None, error: Exception | None = None, available: bool = True):
        from app.kb.parsers.base import PageText

        self.pages = pages if pages is not None else [PageText(1, "apple banana"), PageText(2, "cherry durian")]
        self.error = error
        self.available = available
        self.parsed: list[str] = []
        self.released = 0

    def availability(self):
        return (True, "") if self.available else (False, "未安裝")

    def parse(self, pdf_path, on_progress):
        self.parsed.append(pdf_path)
        if self.error is not None:
            raise self.error
        on_progress(len(self.pages), len(self.pages))
        return self.pages

    def release(self):
        self.released += 1

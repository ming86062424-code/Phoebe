from app.llm import ChatDelta


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
        self.calls: list[dict] = []

    async def list_models(self) -> list[str]:
        return ["fake-model"]

    async def availability(self) -> tuple[bool, str]:
        return True, ""

    async def stream_chat(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        temperature: float | None = None,
    ):
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "system": system,
                "temperature": temperature,
            }
        )
        if self.error is not None and self.error_after is None:
            raise self.error
        for i, delta in enumerate(self.deltas):
            if self.error is not None and i == self.error_after:
                raise self.error
            yield delta
        if self.error is not None and self.error_after is not None and self.error_after >= len(self.deltas):
            raise self.error


class FakeParser:
    id = "fake"
    label = "假解析器"

    def __init__(self, pages=None, error: Exception | None = None, available: bool = True):
        from app.kb.parsers.base import ParsedChunk

        self.pages = pages if pages is not None else [ParsedChunk(1, "apple banana"), ParsedChunk(2, "cherry durian")]
        self.error = error
        self.available = available
        self.parsed: list[str] = []
        self.out_dirs: list = []
        self.released = 0

    def availability(self):
        return (True, "") if self.available else (False, "未安裝")

    def parse(self, path, out_dir, on_progress):
        self.parsed.append(path)
        self.out_dirs.append(out_dir)
        if self.error is not None:
            raise self.error
        on_progress(len(self.pages), len(self.pages))
        return self.pages

    def release(self):
        self.released += 1

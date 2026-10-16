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

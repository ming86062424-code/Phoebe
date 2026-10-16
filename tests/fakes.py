class FakeChatClient:
    def __init__(
        self,
        chunks: list[str] | None = None,
        error: Exception | None = None,
        error_after: int | None = None,
    ):
        self.chunks = ["你", "好"] if chunks is None else chunks
        self.error = error
        self.error_after = error_after
        self.calls: list[list[dict]] = []

    async def stream_chat(self, messages: list[dict]):
        self.calls.append([dict(m) for m in messages])
        if self.error is not None and self.error_after is None:
            raise self.error
        for i, chunk in enumerate(self.chunks):
            if self.error is not None and i == self.error_after:
                raise self.error
            yield chunk
        if self.error is not None and self.error_after is not None and self.error_after >= len(self.chunks):
            raise self.error

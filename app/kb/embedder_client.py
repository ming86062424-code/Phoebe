import base64
import logging
import time
from collections.abc import Callable

import httpx

from app.kb.chunk_store import KnowledgeStoreError

logger = logging.getLogger(__name__)

BATCH_SIZE = 16
TIMEOUT_SECONDS = 120.0
CONNECT_ERROR = "無法連線到 embedding 服務"
LOADING_ERROR = "無法連線到 embedding 服務（模型載入中）"
# Indexing requests are retried 3 times when the embedder is unreachable or still loading
# (e.g. it was killed for lack of memory and is restarting); the waits give it time to load
# the model again. Queries are not retried: a chat answer should not stall for minutes.
RETRY_DELAYS_SECONDS = (10.0, 30.0, 60.0)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


class EmbedderClient:
    """HTTP client of the embedder service; every method returns one vector per input."""

    def __init__(
        self,
        url: str,
        transport: httpx.BaseTransport | None = None,
        retry_delays: tuple[float, ...] = RETRY_DELAYS_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._url = url.rstrip("/")
        self._client = httpx.Client(timeout=TIMEOUT_SECONDS, transport=transport)
        self._retry_delays = retry_delays
        self._sleep = sleep

    def _post(self, batch: list[dict], retry: bool) -> httpx.Response:
        """POST one batch; with `retry`, an unreachable or loading embedder is tried again
        after each of the retry delays before the request fails."""
        delays = self._retry_delays if retry else ()
        for attempt in range(len(delays) + 1):
            try:
                response = self._client.post(f"{self._url}/embed", json={"items": batch})
            except httpx.HTTPError as exc:
                error, cause = CONNECT_ERROR, exc
            else:
                if response.status_code != 503:
                    return response
                error, cause = LOADING_ERROR, None
            if attempt == len(delays):
                if delays:
                    error = f"{error}，已重試 {len(delays)} 次"
                raise KnowledgeStoreError(error) from cause
            logger.warning(
                "%s，%.0f 秒後重試（第 %d／%d 次）", error, delays[attempt], attempt + 1, len(delays)
            )
            self._sleep(delays[attempt])
        raise AssertionError("unreachable")

    def _embed(self, items: list[dict], retry: bool = False) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(items), BATCH_SIZE):
            batch = items[start : start + BATCH_SIZE]
            response = self._post(batch, retry)
            if response.status_code != 200:
                raise KnowledgeStoreError(f"embedding 服務回應錯誤 {response.status_code}")
            try:
                embeddings = response.json()["embeddings"]
            except (ValueError, KeyError, TypeError) as exc:
                raise KnowledgeStoreError("embedding 服務回傳無法解析的資料") from exc
            if not isinstance(embeddings, list):
                raise KnowledgeStoreError("embedding 服務回傳無法解析的資料")
            if len(embeddings) != len(batch):
                raise KnowledgeStoreError("embedding 服務回傳的向量數量不符")
            vectors.extend(embeddings)
        return vectors

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed([{"kind": "query", "text": t} for t in texts])

    # documents, images and page images are embedded while indexing: retried

    def embed_documents(self, title: str, texts: list[str]) -> list[list[float]]:
        items = [{"kind": "document", "title": title, "text": t} for t in texts]
        return self._embed(items, retry=True)

    def embed_images(self, images: list[bytes]) -> list[list[float]]:
        return self._embed([{"kind": "image", "image_b64": _b64(i)} for i in images], retry=True)

    def embed_image_texts(self, items: list[tuple[bytes, str]]) -> list[list[float]]:
        return self._embed(
            [{"kind": "image_text", "image_b64": _b64(i), "text": t} for i, t in items],
            retry=True,
        )

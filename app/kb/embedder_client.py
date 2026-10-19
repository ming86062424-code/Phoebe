import base64

import httpx

from app.kb.chunk_store import KnowledgeStoreError

BATCH_SIZE = 16
TIMEOUT_SECONDS = 120.0
CONNECT_ERROR = "無法連線到 embedding 服務"
LOADING_ERROR = "無法連線到 embedding 服務（模型載入中）"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


class EmbedderClient:
    """HTTP client of the embedder service; every method returns one vector per input."""

    def __init__(self, url: str, transport: httpx.BaseTransport | None = None):
        self._url = url.rstrip("/")
        self._client = httpx.Client(timeout=TIMEOUT_SECONDS, transport=transport)

    def _embed(self, items: list[dict]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(items), BATCH_SIZE):
            batch = items[start : start + BATCH_SIZE]
            try:
                response = self._client.post(f"{self._url}/embed", json={"items": batch})
            except httpx.HTTPError as exc:
                raise KnowledgeStoreError(CONNECT_ERROR) from exc
            if response.status_code == 503:
                raise KnowledgeStoreError(LOADING_ERROR)
            if response.status_code != 200:
                raise KnowledgeStoreError(f"embedding 服務回應錯誤 {response.status_code}")
            try:
                embeddings = response.json()["embeddings"]
            except (ValueError, KeyError, TypeError) as exc:
                raise KnowledgeStoreError("embedding 服務回傳無法解析的資料") from exc
            if len(embeddings) != len(batch):
                raise KnowledgeStoreError("embedding 服務回傳的向量數量不符")
            vectors.extend(embeddings)
        return vectors

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed([{"kind": "query", "text": t} for t in texts])

    def embed_documents(self, title: str, texts: list[str]) -> list[list[float]]:
        return self._embed([{"kind": "document", "title": title, "text": t} for t in texts])

    def embed_images(self, images: list[bytes]) -> list[list[float]]:
        return self._embed([{"kind": "image", "image_b64": _b64(i)} for i in images])

    def embed_image_texts(self, items: list[tuple[bytes, str]]) -> list[list[float]]:
        return self._embed(
            [{"kind": "image_text", "image_b64": _b64(i), "text": t} for i, t in items]
        )

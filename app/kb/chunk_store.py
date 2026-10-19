from dataclasses import dataclass
from typing import Protocol


class KnowledgeStoreError(Exception):
    """The chunk store can't be reached or rejected the request; the message is user-facing."""


@dataclass(frozen=True)
class ChunkRecord:
    kb_id: int
    document_id: int
    filename: str
    page: int
    chunk_index: int
    text: str
    dense: list[float]


@dataclass(frozen=True)
class Hit:
    chunk_id: str  # "{document_id}:{chunk_index}"
    kb_id: int
    document_id: int
    filename: str
    page: int
    text: str
    score: float  # cosine for dense_search, BM25 for keyword_search


class ChunkStore(Protocol):
    def ensure_ready(self, dim: int) -> None: ...

    def insert(self, records: list[ChunkRecord]) -> None: ...

    def delete_document(self, kb_id: int, document_id: int) -> None: ...

    def delete_kb(self, kb_id: int) -> None: ...

    def dense_search(self, kb_id: int, vector: list[float], limit: int) -> list[Hit]: ...

    def keyword_search(self, kb_id: int, query: str, limit: int) -> list[Hit]: ...

import math
import re

from app.kb.chunk_store import ChunkRecord, Hit


def _tokens(text: str) -> set[str]:
    """Whitespace words plus characters and bigrams of CJK runs; stands in for the analyzer."""
    text = text.lower()
    tokens = set(text.split())
    for run in re.findall(r"[一-鿿]+", text):
        tokens.update(run[i : i + 2] for i in range(len(run) - 1))
        tokens.update(run)
    return tokens


def _cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=False)) / (na * nb)


class InMemoryChunkStore:
    """ChunkStore for unit tests: cosine for dense, token overlap for keyword."""

    def __init__(self):
        self.rows: dict[str, ChunkRecord] = {}
        self.dim: int | None = None

    def ensure_ready(self, dim: int) -> None:
        self.dim = dim

    def insert(self, records: list[ChunkRecord]) -> None:
        for r in records:
            self.rows[f"{r.document_id}:{r.chunk_index}"] = r

    def delete_document(self, kb_id: int, document_id: int) -> None:
        self.rows = {
            k: r
            for k, r in self.rows.items()
            if not (r.kb_id == kb_id and r.document_id == document_id)
        }

    def delete_kb(self, kb_id: int) -> None:
        self.rows = {k: r for k, r in self.rows.items() if r.kb_id != kb_id}

    def _hit(self, chunk_id: str, r: ChunkRecord, score: float) -> Hit:
        return Hit(chunk_id, r.kb_id, r.document_id, r.filename, r.page, r.text, score)

    def dense_search(self, kb_id: int, vector: list[float], limit: int) -> list[Hit]:
        scored = [
            self._hit(k, r, _cosine(vector, r.dense))
            for k, r in self.rows.items()
            if r.kb_id == kb_id
        ]
        return sorted(scored, key=lambda h: h.score, reverse=True)[:limit]

    def keyword_search(self, kb_id: int, query: str, limit: int) -> list[Hit]:
        wanted = _tokens(query)
        scored = []
        for k, r in self.rows.items():
            if r.kb_id != kb_id:
                continue
            overlap = len(wanted & _tokens(r.text))
            if overlap:
                scored.append(self._hit(k, r, float(overlap)))
        return sorted(scored, key=lambda h: h.score, reverse=True)[:limit]

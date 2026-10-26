import logging
import math
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from app.kb.chunk_store import ChunkRecord, ChunkStore, Hit
from app.kb.parsers.base import ImageRef, ParsedChunk

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 8192
CANDIDATE_FACTOR = 4  # each retrieval path returns top_k * 4 candidates for fusion
DEFAULT_VECTOR_WEIGHT = 0.5  # KB_VECTOR_WEIGHT; the keyword path gets the rest
# Page images per embedding request. The embedder runs a whole request in one forward
# pass, so this is also how many page images (each with its page text) it holds at once:
# 8 at a time peaked at ~4.7 GB of VRAM on the 6 GB RTX 2060 in GPU mode, 2 needs ~1 GiB
# less for ~6 % more time (docs/gpu-memory.md). One number for CPU and GPU mode.
IMAGE_BATCH_SIZE = 2


class Embedder(Protocol):
    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...

    def embed_documents(self, title: str, texts: list[str]) -> list[list[float]]: ...

    def embed_images(self, images: list[bytes]) -> list[list[float]]: ...

    def embed_image_texts(self, items: list[tuple[bytes, str]]) -> list[list[float]]: ...


def _image_dict(ref: ImageRef) -> dict:
    return {"name": ref.name, "width": ref.width, "height": ref.height}


@dataclass(frozen=True)
class Source:
    document_id: int
    filename: str
    page: int  # 0 for text files
    score: float  # weighted hybrid score, 0..1
    text: str  # the whole chunk: for the model, the UI and the saved message
    vector_score: float | None = None  # cosine, None when the dense path missed it
    keyword_score: float | None = None  # BM25, None when the keyword path missed it
    kind: str = "text"  # "text" or "page_image"
    images: list[dict] = field(default_factory=list)  # [{"name", "width", "height"}]
    page_image: dict | None = None  # {"name", "width", "height"}

    def to_dict(self) -> dict:
        """The client/database form."""
        return asdict(self)


_IMAGE_MARKER = re.compile(r"!\[([^\]\n]*)\]\(img:[^)\s]+\)")


@dataclass(frozen=True)
class Chunk:
    """One stored chunk of a document, as the chunk viewer shows it."""

    index: int  # 1-based position in the document
    page: int  # 0 for text files
    kind: str  # "text" or "page_image"
    text: str
    images: list[dict] = field(default_factory=list)  # [{"name", "width", "height"}]
    page_image: dict | None = None

    def to_dict(self) -> dict:
        # an image marker ![圖片 1](img:NAME) counts as its label, not as markup
        return {**asdict(self), "chars": len(_IMAGE_MARKER.sub(r"\1", self.text))}


def normalize_cosine(score: float) -> float:
    """Milvus WeightedRanker (norm_score) for COSINE: [-1, 1] -> [0, 1]."""
    return (1 + score) * 0.5


def normalize_bm25(score: float) -> float:
    """Milvus WeightedRanker (norm_score) for BM25: [0, inf) -> [0, 1)."""
    return 2 * math.atan(score) / math.pi


def weighted_fuse(
    dense: list[Hit], keyword: list[Hit], top_k: int, vector_weight: float
) -> list[tuple[Hit, float, float | None, float | None]]:
    """Weighted hybrid score, as Milvus' WeightedRanker with norm_score (v2.6.25
    internal/util/function/rerank/util.go):

        score = w * normalize_cosine(cosine) + (1 - w) * normalize_bm25(bm25)

    A path that did not return the hit contributes 0. Returns
    (hit, score, cosine or None, bm25 or None), best first.
    """
    if not 0 <= vector_weight <= 1:
        raise ValueError("向量權重必須介於 0 到 1")
    keyword_weight = 1 - vector_weight
    hits: dict[str, Hit] = {}
    vector_scores: dict[str, float] = {}
    keyword_scores: dict[str, float] = {}
    for hit in dense:
        hits.setdefault(hit.chunk_id, hit)
        vector_scores.setdefault(hit.chunk_id, hit.score)
    for hit in keyword:
        hits.setdefault(hit.chunk_id, hit)
        keyword_scores.setdefault(hit.chunk_id, hit.score)

    def score(cid: str) -> float:
        total = 0.0
        if cid in vector_scores:
            total += vector_weight * normalize_cosine(vector_scores[cid])
        if cid in keyword_scores:
            total += keyword_weight * normalize_bm25(keyword_scores[cid])
        return total

    fused = {cid: score(cid) for cid in hits}
    # sorted() is stable, so ties keep first-seen order (dense list first)
    ranked = sorted(fused, key=lambda cid: fused[cid], reverse=True)[: max(top_k, 0)]
    return [
        (hits[cid], fused[cid], vector_scores.get(cid), keyword_scores.get(cid)) for cid in ranked
    ]


class KnowledgeIndex:
    """Embeds and retrieves parsed chunks; hybrid dense + BM25 search via a ChunkStore."""

    def __init__(self, store: ChunkStore, embedder: Embedder):
        self._store = store
        self._embedder = embedder
        self._ready = False
        self._ready_lock = threading.Lock()

    def _ensure_ready(self, dim: int) -> None:
        """Create or validate the store once; the dimension comes from a real vector, so
        no extra probe call is needed. A failure is not cached, the next call retries."""
        if self._ready:
            return
        with self._ready_lock:
            if not self._ready:
                self._store.ensure_ready(dim)
                self._ready = True

    def add_document(
        self,
        kb_id: int,
        document_id: int,
        filename: str,
        chunks: list[ParsedChunk],
        image_dir: Path | None = None,
        timings: dict[str, float] | None = None,
    ) -> int:
        """Index the chunks as they are (one per page) and return how many were written.

        Replaces any chunks the document already has, so repeating it is harmless.
        "page_image" chunks read `image_dir / chunk.page_image.name`.
        `timings`, when given, receives the seconds spent on "embed" and "store".
        """
        kept = [c for c in chunks if c.kind == "page_image" or c.text.strip()]
        if not kept:
            self._store.delete_document(kb_id, document_id)
            return 0
        # Embedding can take minutes for a big PDF; no lock is held.
        started = time.perf_counter()
        embeddings = self._embed(filename, kept, image_dir)
        embedded = time.perf_counter()
        self._ensure_ready(len(embeddings[0]))
        records = []
        for i, (chunk, embedding) in enumerate(zip(kept, embeddings, strict=True)):
            text = chunk.text
            if len(text) > MAX_TEXT_CHARS:
                logger.warning(
                    "chunk %s:%s has %d characters; truncated to %d",
                    document_id,
                    i,
                    len(text),
                    MAX_TEXT_CHARS,
                )
                text = text[:MAX_TEXT_CHARS]
            records.append(
                ChunkRecord(
                    kb_id=kb_id,
                    document_id=document_id,
                    filename=filename,
                    page=int(chunk.page),
                    chunk_index=i,
                    text=text,
                    dense=[float(x) for x in embedding],
                    kind=chunk.kind,
                    images=list(chunk.images),
                    page_image=chunk.page_image,
                )
            )
        # a retry must not leave the previous attempt's chunks behind
        self._store.delete_document(kb_id, document_id)
        self._store.insert(records)
        if timings is not None:
            timings["embed"] = embedded - started
            timings["store"] = time.perf_counter() - embedded
        return len(records)

    def _embed(
        self, filename: str, chunks: list[ParsedChunk], image_dir: Path | None
    ) -> list[list[float]]:
        """One vector per chunk, in order: text chunks as documents (the embedder adds the
        `title: ... | text: ...` prefix), page_image chunks as image + text."""
        vectors: list[list[float] | None] = [None] * len(chunks)
        text_at = [i for i, c in enumerate(chunks) if c.kind != "page_image"]
        image_at = [i for i, c in enumerate(chunks) if c.kind == "page_image"]
        if text_at:
            found = self._embedder.embed_documents(filename, [chunks[i].text for i in text_at])
            for i, vector in zip(text_at, found, strict=True):
                vectors[i] = vector
        if image_at and image_dir is None:
            raise ValueError("page_image chunks need image_dir")
        for start in range(0, len(image_at), IMAGE_BATCH_SIZE):
            batch = image_at[start : start + IMAGE_BATCH_SIZE]
            items = []
            for i in batch:
                ref = chunks[i].page_image
                if ref is None:
                    raise ValueError(f"page {chunks[i].page} is a page_image chunk without an image")
                items.append(((image_dir / ref.name).read_bytes(), chunks[i].text))
            found = self._embedder.embed_image_texts(items)
            for i, vector in zip(batch, found, strict=True):
                vectors[i] = vector
        return vectors  # type: ignore[return-value]

    def list_chunks(self, kb_id: int, document_id: int) -> list[Chunk]:
        """The document's chunks exactly as stored (and as the model would get them)."""
        hits = self._store.list_chunks(kb_id, document_id)
        return [
            Chunk(
                index=i,
                page=hit.page,
                kind=hit.kind,
                text=hit.text,
                images=[_image_dict(ref) for ref in hit.images],
                page_image=None if hit.page_image is None else _image_dict(hit.page_image),
            )
            for i, hit in enumerate(hits, start=1)
        ]

    def delete_document(self, kb_id: int, document_id: int) -> None:
        self._store.delete_document(kb_id, document_id)

    def delete_kb(self, kb_id: int) -> None:
        self._store.delete_kb(kb_id)

    def search(
        self,
        kb_id: int,
        query: str,
        top_k: int,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        timings: dict[str, float] | None = None,
    ) -> list[Source]:
        """Hybrid search; `vector_weight` (0..1) is the dense path's share of the score.
        `timings`, when given, receives the seconds spent on "embed" and "search"."""
        if top_k <= 0:
            return []
        started = time.perf_counter()
        vector = [float(x) for x in self._embedder.embed_queries([query])[0]]
        embedded = time.perf_counter()
        self._ensure_ready(len(vector))
        limit = top_k * CANDIDATE_FACTOR
        dense = self._store.dense_search(kb_id, vector, limit)
        keyword = self._store.keyword_search(kb_id, query, limit)
        if timings is not None:
            timings["embed"] = embedded - started
            timings["search"] = time.perf_counter() - embedded
        return [
            Source(
                document_id=hit.document_id,
                filename=hit.filename,
                page=hit.page,
                score=round(fused, 4),
                text=hit.text,
                vector_score=None if vec is None else round(vec, 4),
                keyword_score=None if kw is None else round(kw, 4),
                kind=hit.kind,
                images=[_image_dict(ref) for ref in hit.images],
                page_image=None if hit.page_image is None else _image_dict(hit.page_image),
            )
            for hit, fused, vec, kw in weighted_fuse(dense, keyword, top_k, vector_weight)
        ]

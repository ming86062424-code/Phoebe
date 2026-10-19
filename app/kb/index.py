import logging
import threading
from dataclasses import asdict, dataclass

from llama_index.core import Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, MetadataMode, TextNode

from app.kb.chunk_store import ChunkRecord, ChunkStore, Hit
from app.kb.parsers.base import PageText

logger = logging.getLogger(__name__)

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SNIPPET_CHARS = 400
MAX_TEXT_CHARS = 8192
RRF_K = 60
CANDIDATE_FACTOR = 4  # each retrieval path returns top_k * 4 candidates for fusion


@dataclass(frozen=True)
class Source:
    document_id: int
    filename: str
    page: int
    score: float  # RRF score
    text: str  # short snippet for the UI and the saved message
    content: str  # the whole chunk, for the model
    vector_score: float | None = None  # cosine, None when the dense path missed it
    keyword_score: float | None = None  # BM25, None when the keyword path missed it

    def to_dict(self) -> dict:
        """The client/database form: everything but the full chunk."""
        data = asdict(self)
        del data["content"]
        return data


def rrf_fuse(
    dense: list[Hit], keyword: list[Hit], top_k: int
) -> list[tuple[Hit, float, float | None, float | None]]:
    """Reciprocal rank fusion: score = sum of 1 / (60 + rank), rank starting at 1.

    Returns (hit, rrf score, cosine or None, bm25 or None), best first.
    """
    fused: dict[str, float] = {}
    hits: dict[str, Hit] = {}
    vector_scores: dict[str, float] = {}
    keyword_scores: dict[str, float] = {}
    for rank, hit in enumerate(dense, start=1):
        fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        hits.setdefault(hit.chunk_id, hit)
        vector_scores.setdefault(hit.chunk_id, hit.score)
    for rank, hit in enumerate(keyword, start=1):
        fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        hits.setdefault(hit.chunk_id, hit)
        keyword_scores.setdefault(hit.chunk_id, hit.score)
    # sorted() is stable, so ties keep first-seen order (dense list first)
    ranked = sorted(fused, key=lambda cid: fused[cid], reverse=True)[: max(top_k, 0)]
    return [
        (hits[cid], fused[cid], vector_scores.get(cid), keyword_scores.get(cid)) for cid in ranked
    ]


class KnowledgeIndex:
    """Chunks, embeds and retrieves document pages; hybrid dense + BM25 search via a ChunkStore."""

    def __init__(self, store: ChunkStore, embed_model: BaseEmbedding):
        self._store = store
        self._embed_model = embed_model
        self._splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
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
        self, kb_id: int, document_id: int, filename: str, pages: list[PageText]
    ) -> int:
        """Index the pages and return the number of chunks written."""
        docs = [
            Document(
                text=p.text,
                metadata={"kb_document_id": document_id, "filename": filename, "page": p.page},
                excluded_embed_metadata_keys=["kb_document_id", "page"],
                excluded_llm_metadata_keys=["kb_document_id"],
            )
            for p in pages
            if p.text.strip()
        ]
        nodes = self._splitter.get_nodes_from_documents(docs)
        return self._write_nodes(kb_id, document_id, filename, nodes)

    def add_chunks(
        self, kb_id: int, document_id: int, filename: str, chunks: list[tuple[int, str]]
    ) -> int:
        """Index ready-made (page, text) chunks without splitting them again.

        Replaces any chunks the document already has, so repeating it is harmless.
        """
        nodes = [
            TextNode(
                text=text,
                metadata={"kb_document_id": document_id, "filename": filename, "page": page},
                excluded_embed_metadata_keys=["kb_document_id", "page"],
                excluded_llm_metadata_keys=["kb_document_id"],
            )
            for page, text in chunks
            if text.strip()
        ]
        if not nodes:
            self._store.delete_document(kb_id, document_id)
            return 0
        return self._write_nodes(kb_id, document_id, filename, nodes)

    def _write_nodes(
        self, kb_id: int, document_id: int, filename: str, nodes: list[BaseNode]
    ) -> int:
        if not nodes:
            return 0
        # Embedding calls Ollama and can take minutes for a big PDF; no lock is held.
        embeddings = self._embed_model.get_text_embedding_batch(
            [n.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes]
        )
        self._ensure_ready(len(embeddings[0]))
        records = []
        for i, (node, embedding) in enumerate(zip(nodes, embeddings, strict=True)):
            text = node.get_content()
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
                    page=int(node.metadata["page"]),
                    chunk_index=i,
                    text=text,
                    dense=[float(x) for x in embedding],
                )
            )
        # a retry must not leave the previous attempt's chunks behind
        self._store.delete_document(kb_id, document_id)
        self._store.insert(records)
        return len(records)

    def delete_document(self, kb_id: int, document_id: int) -> None:
        self._store.delete_document(kb_id, document_id)

    def delete_kb(self, kb_id: int) -> None:
        self._store.delete_kb(kb_id)

    def search(self, kb_id: int, query: str, top_k: int) -> list[Source]:
        if top_k <= 0:
            return []
        vector = [float(x) for x in self._embed_model.get_query_embedding(query)]
        self._ensure_ready(len(vector))
        limit = top_k * CANDIDATE_FACTOR
        dense = self._store.dense_search(kb_id, vector, limit)
        keyword = self._store.keyword_search(kb_id, query, limit)
        return [
            Source(
                document_id=hit.document_id,
                filename=hit.filename,
                page=hit.page,
                score=round(rrf, 4),
                text=hit.text[:SNIPPET_CHARS],
                content=hit.text,
                vector_score=None if vec is None else round(vec, 4),
                keyword_score=None if kw is None else round(kw, 4),
            )
            for hit, rrf, vec, kw in rrf_fuse(dense, keyword, top_k)
        ]

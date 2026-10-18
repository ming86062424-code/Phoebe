import threading
from dataclasses import asdict, dataclass

import chromadb
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.kb.parsers.base import PageText

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SNIPPET_CHARS = 400


@dataclass(frozen=True)
class Source:
    document_id: int
    filename: str
    page: int
    score: float
    text: str  # short snippet for the UI and the saved message
    content: str  # the whole chunk, for the model

    def to_dict(self) -> dict:
        """The client/database form: everything but the full chunk."""
        data = asdict(self)
        del data["content"]
        return data


def _collection_name(kb_id: int) -> str:
    return f"kb_{kb_id}"


class KnowledgeIndex:
    """Chunks, embeds and retrieves document pages; one Chroma collection per knowledge base."""

    def __init__(self, chroma_dir: str, embed_model: BaseEmbedding):
        self._client = chromadb.PersistentClient(
            path=chroma_dir, settings=chromadb.Settings(anonymized_telemetry=False)
        )
        self._embed_model = embed_model
        self._splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        # Chroma's local client isn't safe for concurrent writes from several threads.
        self._lock = threading.Lock()

    def _index(self, kb_id: int) -> tuple[VectorStoreIndex, chromadb.Collection]:
        collection = self._client.get_or_create_collection(
            _collection_name(kb_id), metadata={"hnsw:space": "cosine"}
        )
        store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(store, embed_model=self._embed_model)
        return index, collection

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
        if not nodes:
            return 0
        with self._lock:
            index, _ = self._index(kb_id)
            index.insert_nodes(nodes)
        return len(nodes)

    def delete_document(self, kb_id: int, document_id: int) -> None:
        with self._lock:
            _, collection = self._index(kb_id)
            collection.delete(where={"kb_document_id": document_id})

    def delete_kb(self, kb_id: int) -> None:
        with self._lock:
            try:
                self._client.delete_collection(_collection_name(kb_id))
            except Exception:
                pass  # never indexed anything

    def search(self, kb_id: int, query: str, top_k: int) -> list[Source]:
        with self._lock:
            index, collection = self._index(kb_id)
            if collection.count() == 0:
                return []
        results = index.as_retriever(similarity_top_k=top_k).retrieve(query)
        return [
            Source(
                document_id=int(r.node.metadata["kb_document_id"]),
                filename=str(r.node.metadata["filename"]),
                page=int(r.node.metadata["page"]),
                score=round(float(r.score or 0.0), 4),
                text=r.node.get_content()[:SNIPPET_CHARS],
                content=r.node.get_content(),
            )
            for r in results
        ]

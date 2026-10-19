import threading

import pytest

from app.kb import index as index_module
from app.kb.chunk_store import Hit, KnowledgeStoreError
from app.kb.index import KnowledgeIndex, Source, rrf_fuse
from app.kb.parsers.base import PageText
from tests.fakes import KeywordEmbedding
from tests.fakes_store import InMemoryChunkStore


@pytest.fixture
def store():
    return InMemoryChunkStore()


@pytest.fixture
def index(store):
    return KnowledgeIndex(store, KeywordEmbedding())


def hit(cid, score=1.0, text="t"):
    doc, _, chunk = cid.partition(":")
    return Hit(cid, 1, int(doc), "a.pdf", 1, text + chunk, score)


# ---- rrf_fuse ----


def test_rrf_ranks_hits_in_both_lists_first():
    dense = [hit("1:0", 0.9), hit("2:0", 0.8), hit("3:0", 0.7)]
    keyword = [hit("3:0", 5.0), hit("1:0", 4.0)]
    fused = rrf_fuse(dense, keyword, top_k=3)
    assert [h.chunk_id for h, *_ in fused] == ["1:0", "3:0", "2:0"]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_keyword_only_hit_is_returned():
    fused = rrf_fuse([hit("1:0", 0.9)], [hit("9:0", 3.0)], top_k=5)
    by_id = {h.chunk_id: (rrf, vec, kw) for h, rrf, vec, kw in fused}
    assert by_id["9:0"] == (pytest.approx(1 / 61), None, 3.0)
    assert by_id["1:0"][2] is None


def test_scores_are_carried():
    fused = rrf_fuse([hit("1:0", 0.9)], [hit("1:0", 7.5)], top_k=1)
    h, rrf, vec, kw = fused[0]
    assert (vec, kw) == (0.9, 7.5)
    assert rrf == pytest.approx(2 / 61)


def test_top_k_and_empty():
    dense = [hit(f"{i}:0") for i in range(1, 6)]
    assert len(rrf_fuse(dense, [], top_k=2)) == 2
    assert rrf_fuse([], [], top_k=3) == []
    assert rrf_fuse(dense, [], top_k=0) == []


def test_search_returns_keyword_only_match(store):
    """A code the embedding can't link to the query is still found through the keyword path."""

    class FarQueryEmbedding(KeywordEmbedding):
        def _get_query_embedding(self, query):
            v = [0.0] * self.dim
            v[0] = 1.0
            return v

        def _get_text_embedding(self, text):
            v = [0.0] * self.dim
            v[1] = 1.0  # orthogonal to every query: cosine 0
            return v

    class NegativeDense(InMemoryChunkStore):
        def dense_search(self, kb_id, vector, limit):
            return []  # the dense path finds nothing at all

    index = KnowledgeIndex(NegativeDense(), FarQueryEmbedding())
    index.add_document(1, 10, "a.pdf", [PageText(1, "Invoice No. 2026-1018 total due")])
    results = index.search(1, "2026-1018", top_k=3)
    assert results and results[0].document_id == 10
    assert results[0].vector_score is None
    assert results[0].keyword_score is not None


def test_search_with_non_positive_top_k_returns_nothing(index):
    index.add_document(1, 10, "a.pdf", [PageText(1, "apple")])
    assert index.search(1, "apple", top_k=0) == []
    assert index.search(1, "apple", top_k=-1) == []


def test_dimension_mismatch_propagates(store):
    class Small(KeywordEmbedding):
        dim: int = 8

    KnowledgeIndex(store, KeywordEmbedding()).search(1, "x", top_k=1)
    with pytest.raises(KnowledgeStoreError, match=r"向量維度（8）與既有知識庫（64）不同，請重建知識庫"):
        KnowledgeIndex(store, Small()).search(1, "x", top_k=1)


# ---- KnowledgeIndex ----


def test_search_returns_best_page_with_metadata(index):
    pages = [PageText(1, "apple banana orchard"), PageText(2, "rocket engine fuel")]
    assert index.add_document(1, 10, "a.pdf", pages) == 2
    results = index.search(1, "rocket fuel", top_k=1)
    assert len(results) == 1
    assert (results[0].document_id, results[0].filename, results[0].page) == (10, "a.pdf", 2)
    assert "rocket" in results[0].text
    assert results[0].vector_score is not None


def test_search_empty_kb_returns_nothing(index):
    assert index.search(99, "anything", top_k=3) == []


def test_kbs_are_isolated(index):
    index.add_document(1, 10, "a.pdf", [PageText(1, "apple")])
    index.add_document(2, 20, "b.pdf", [PageText(1, "apple")])
    assert {r.document_id for r in index.search(1, "apple", top_k=5)} == {10}


def test_reindex_replaces_old_chunks(index, store):
    pages = [PageText(1, "apple banana"), PageText(2, "rocket fuel")]
    assert index.add_document(1, 10, "a.pdf", pages) == 2
    assert index.add_document(1, 10, "a.pdf", pages[:1]) == 1
    assert len(store.rows) == 1
    assert [r.document_id for r in index.search(1, "rocket", top_k=5)] == [10]
    assert all("rocket" not in r.text for r in index.search(1, "rocket", top_k=5))


def test_delete_document_and_kb(index):
    index.add_document(1, 10, "a.pdf", [PageText(1, "apple")])
    index.add_document(1, 11, "b.pdf", [PageText(1, "apple pie")])
    index.delete_document(1, 10)
    assert {r.document_id for r in index.search(1, "apple", top_k=5)} == {11}
    index.delete_kb(1)
    assert index.search(1, "apple", top_k=5) == []
    index.delete_kb(1)  # idempotent


def test_blank_pages_are_skipped(index):
    assert index.add_document(1, 10, "a.pdf", [PageText(1, "  ")]) == 0


def test_long_text_is_truncated_to_8192(store, monkeypatch):
    assert index_module.MAX_TEXT_CHARS == 8192
    monkeypatch.setattr(index_module, "MAX_TEXT_CHARS", 20)
    index = KnowledgeIndex(store, KeywordEmbedding())
    index.add_document(1, 10, "a.pdf", [PageText(1, "word " * 50)])
    assert store.rows and all(len(r.text) <= 20 for r in store.rows.values())


def test_snippet_is_short_but_content_is_whole_chunk(store):
    index = KnowledgeIndex(store, KeywordEmbedding())
    text = "alpha " * 100  # about 600 characters, a single chunk
    index.add_document(1, 10, "a.pdf", [PageText(1, text)])
    (source,) = index.search(1, "alpha", top_k=1)
    assert len(source.text) == index_module.SNIPPET_CHARS
    assert source.content == store.rows["10:0"].text and len(source.content) > len(source.text)


def test_source_to_dict_includes_component_scores():
    data = Source(1, "a.pdf", 2, 0.0328, "snip", "full", 0.9, None).to_dict()
    assert data == {
        "document_id": 1,
        "filename": "a.pdf",
        "page": 2,
        "score": 0.0328,
        "text": "snip",
        "vector_score": 0.9,
        "keyword_score": None,
    }


def test_dimension_is_checked_once_lazily(store):
    index = KnowledgeIndex(store, KeywordEmbedding())
    assert store.dim is None  # nothing at construction
    index.search(1, "x", top_k=1)
    assert store.dim == 64


def test_store_error_surfaces_and_is_retried():
    calls = []

    class Flaky(InMemoryChunkStore):
        def ensure_ready(self, dim):
            calls.append(dim)
            if len(calls) == 1:
                raise KnowledgeStoreError("無法連線到 Milvus")
            super().ensure_ready(dim)

    index = KnowledgeIndex(Flaky(), KeywordEmbedding())
    with pytest.raises(KnowledgeStoreError, match="無法連線到 Milvus"):
        index.search(1, "x", top_k=1)
    assert index.search(1, "x", top_k=1) == []
    assert len(calls) == 2


def test_search_is_not_blocked_while_another_document_is_embedding(store):
    started, release = threading.Event(), threading.Event()

    class SlowTextEmbedding(KeywordEmbedding):
        def _get_text_embedding(self, text):
            if "slow" in text:
                started.set()
                release.wait(10)
            return super()._get_text_embedding(text)

        def _get_text_embeddings(self, texts):
            return [self._get_text_embedding(t) for t in texts]

    index = KnowledgeIndex(store, SlowTextEmbedding())
    index.add_document(1, 10, "a.pdf", [PageText(1, "apple banana")])
    writer = threading.Thread(
        target=index.add_document, args=(1, 11, "b.pdf", [PageText(1, "slow rocket")])
    )
    writer.start()
    try:
        assert started.wait(5)
        found: list = []
        reader = threading.Thread(target=lambda: found.extend(index.search(1, "apple", 1)))
        reader.start()
        reader.join(3)
        assert not reader.is_alive(), "search waited for the embedding to finish"
        assert found[0].document_id == 10
    finally:
        release.set()
        writer.join(10)
    assert {r.document_id for r in index.search(1, "rocket", 2)} == {10, 11}


# ---- real Milvus (skipped unless MILVUS_URI is reachable) ----


def _milvus_uri() -> str:
    import os
    import socket
    from urllib.parse import urlparse

    uri = os.environ.get("MILVUS_URI", "http://localhost:19530")
    parsed = urlparse(uri)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 19530), timeout=1):
            return uri
    except OSError:
        pytest.skip(f"Milvus not reachable at {uri}")


@pytest.mark.milvus
def test_real_milvus_finds_invoice_number_through_keyword_path():
    import os

    from pymilvus import MilvusClient

    from app.kb.milvus_store import MilvusChunkStore

    uri, token = _milvus_uri(), os.environ.get("MILVUS_TOKEN", "")
    name = f"phoebe_idx_test_{os.getpid()}"
    client = MilvusClient(uri=uri, token=token)
    index = KnowledgeIndex(MilvusChunkStore(uri, token, name, client=client), KeywordEmbedding())
    try:
        pages = [
            PageText(1, "Invoice No. 2026-1018 issued to ACME, total due 30 days."),
            PageText(2, "Quarterly rocket engine fuel report."),
        ]
        index.add_document(1, 10, "inv.pdf", pages)
        results = index.search(1, "2026-1018", top_k=3)
        assert results and results[0].page == 1
        assert results[0].keyword_score is not None
    finally:
        client.drop_collection(name)

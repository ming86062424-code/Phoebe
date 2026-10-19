import threading

import pytest

from app.kb import index as index_module
from app.kb.chunk_store import Hit, KnowledgeStoreError
from app.kb.index import KnowledgeIndex, Source, rrf_fuse
from app.kb.parsers.base import ImageRef, ParsedChunk
from tests.fakes_embedder import FakeEmbedder
from tests.fakes_store import InMemoryChunkStore


@pytest.fixture
def store():
    return InMemoryChunkStore()


@pytest.fixture
def index(store):
    return KnowledgeIndex(store, FakeEmbedder())


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

    class FarQueryEmbedding(FakeEmbedder):
        def query_vector(self, text):
            v = [0.0] * self.dim
            v[0] = 1.0
            return v

        def document_vector(self, title, text):
            v = [0.0] * self.dim
            v[1] = 1.0  # orthogonal to every query: cosine 0
            return v

    class NegativeDense(InMemoryChunkStore):
        def dense_search(self, kb_id, vector, limit):
            return []  # the dense path finds nothing at all

    index = KnowledgeIndex(NegativeDense(), FarQueryEmbedding())
    index.add_document(1, 10, "a.pdf", [ParsedChunk(1, "Invoice No. 2026-1018 total due")])
    results = index.search(1, "2026-1018", top_k=3)
    assert results and results[0].document_id == 10
    assert results[0].vector_score is None
    assert results[0].keyword_score is not None


def test_search_with_non_positive_top_k_returns_nothing(index):
    index.add_document(1, 10, "a.pdf", [ParsedChunk(1, "apple")])
    assert index.search(1, "apple", top_k=0) == []
    assert index.search(1, "apple", top_k=-1) == []


def test_dimension_mismatch_propagates(store):
    class Small(FakeEmbedder):
        dim: int = 8

    KnowledgeIndex(store, FakeEmbedder()).search(1, "x", top_k=1)
    with pytest.raises(KnowledgeStoreError, match=r"向量維度（8）與既有知識庫（64）不同，請重建知識庫"):
        KnowledgeIndex(store, Small()).search(1, "x", top_k=1)


# ---- KnowledgeIndex ----


def test_search_returns_best_page_with_metadata(index):
    pages = [ParsedChunk(1, "apple banana orchard"), ParsedChunk(2, "rocket engine fuel")]
    assert index.add_document(1, 10, "a.pdf", pages) == 2
    results = index.search(1, "rocket fuel", top_k=1)
    assert len(results) == 1
    assert (results[0].document_id, results[0].filename, results[0].page) == (10, "a.pdf", 2)
    assert "rocket" in results[0].text
    assert results[0].vector_score is not None


def test_search_empty_kb_returns_nothing(index):
    assert index.search(99, "anything", top_k=3) == []


def test_kbs_are_isolated(index):
    index.add_document(1, 10, "a.pdf", [ParsedChunk(1, "apple")])
    index.add_document(2, 20, "b.pdf", [ParsedChunk(1, "apple")])
    assert {r.document_id for r in index.search(1, "apple", top_k=5)} == {10}


def test_reindex_replaces_old_chunks(index, store):
    pages = [ParsedChunk(1, "apple banana"), ParsedChunk(2, "rocket fuel")]
    assert index.add_document(1, 10, "a.pdf", pages) == 2
    assert index.add_document(1, 10, "a.pdf", pages[:1]) == 1
    assert len(store.rows) == 1
    assert [r.document_id for r in index.search(1, "rocket", top_k=5)] == [10]
    assert all("rocket" not in r.text for r in index.search(1, "rocket", top_k=5))


def test_delete_document_and_kb(index):
    index.add_document(1, 10, "a.pdf", [ParsedChunk(1, "apple")])
    index.add_document(1, 11, "b.pdf", [ParsedChunk(1, "apple pie")])
    index.delete_document(1, 10)
    assert {r.document_id for r in index.search(1, "apple", top_k=5)} == {11}
    index.delete_kb(1)
    assert index.search(1, "apple", top_k=5) == []
    index.delete_kb(1)  # idempotent


def test_blank_pages_are_skipped(index):
    assert index.add_document(1, 10, "a.pdf", [ParsedChunk(1, "  ")]) == 0


def test_long_text_is_truncated_to_8192(store, monkeypatch):
    assert index_module.MAX_TEXT_CHARS == 8192
    monkeypatch.setattr(index_module, "MAX_TEXT_CHARS", 20)
    index = KnowledgeIndex(store, FakeEmbedder())
    index.add_document(1, 10, "a.pdf", [ParsedChunk(1, "word " * 50)])
    assert store.rows and all(len(r.text) <= 20 for r in store.rows.values())


def test_source_text_is_the_whole_chunk(store):
    index = KnowledgeIndex(store, FakeEmbedder())
    text = "alpha " * 100  # about 600 characters
    index.add_document(1, 10, "a.pdf", [ParsedChunk(1, text)])
    (source,) = index.search(1, "alpha", top_k=1)
    assert source.text == store.rows["10:0"].text == text


def test_each_page_is_one_chunk(store):
    index = KnowledgeIndex(store, FakeEmbedder())
    long_page = "word " * 1000  # 5000 characters, still a single chunk
    assert index.add_document(1, 10, "a.pdf", [ParsedChunk(1, long_page), ParsedChunk(2, "x")]) == 2
    rows = sorted(store.rows.values(), key=lambda r: r.chunk_index)
    assert [(r.page, r.chunk_index, r.kind) for r in rows] == [(1, 0, "text"), (2, 1, "text")]


def test_source_to_dict_includes_component_scores():
    data = Source(1, "a.pdf", 2, 0.0328, "full text", 0.9, None).to_dict()
    assert data == {
        "document_id": 1,
        "filename": "a.pdf",
        "page": 2,
        "score": 0.0328,
        "text": "full text",
        "vector_score": 0.9,
        "keyword_score": None,
        "kind": "text",
        "images": [],
        "page_image": None,
    }


def test_search_carries_kind_and_images(store, tmp_path):
    (tmp_path / "page_1.png").write_bytes(b"png-1")
    index = KnowledgeIndex(store, FakeEmbedder())
    chunk = ParsedChunk(
        1, "柴犬 photo", images=[ImageRef("img_1_0.png", 30, 40)],
        page_image=ImageRef("page_1.png", 1536, 1000), kind="page_image",
    )  # fmt: skip
    index.add_document(1, 10, "a.pdf", [chunk], image_dir=tmp_path)
    (source,) = index.search(1, "photo", top_k=1)
    data = source.to_dict()
    assert data["kind"] == "page_image"
    assert data["images"] == [{"name": "img_1_0.png", "width": 30, "height": 40}]
    assert data["page_image"] == {"name": "page_1.png", "width": 1536, "height": 1000}


def test_dimension_is_checked_once_lazily(store):
    index = KnowledgeIndex(store, FakeEmbedder())
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

    index = KnowledgeIndex(Flaky(), FakeEmbedder())
    with pytest.raises(KnowledgeStoreError, match="無法連線到 Milvus"):
        index.search(1, "x", top_k=1)
    assert index.search(1, "x", top_k=1) == []
    assert len(calls) == 2


def test_search_is_not_blocked_while_another_document_is_embedding(store):
    started, release = threading.Event(), threading.Event()

    class SlowTextEmbedding(FakeEmbedder):
        def document_vector(self, title, text):
            if "slow" in text:
                started.set()
                release.wait(10)
            return super().document_vector(title, text)

    index = KnowledgeIndex(store, SlowTextEmbedding())
    index.add_document(1, 10, "a.pdf", [ParsedChunk(1, "apple banana")])
    writer = threading.Thread(
        target=index.add_document, args=(1, 11, "b.pdf", [ParsedChunk(1, "slow rocket")])
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
    index = KnowledgeIndex(MilvusChunkStore(uri, token, name, client=client), FakeEmbedder())
    try:
        pages = [
            ParsedChunk(1, "Invoice No. 2026-1018 issued to ACME, total due 30 days."),
            ParsedChunk(2, "Quarterly rocket engine fuel report."),
        ]
        index.add_document(1, 10, "inv.pdf", pages)
        results = index.search(1, "2026-1018", top_k=3)
        assert results and results[0].page == 1
        assert results[0].keyword_score is not None
    finally:
        client.drop_collection(name)


# ---- the embedder is called with the right shapes ----


def test_documents_use_filename_as_title_and_plain_text(store):
    calls = []

    class Spy(FakeEmbedder):
        def embed_documents(self, title, texts):
            calls.append((title, texts))
            return super().embed_documents(title, texts)

        def embed_queries(self, texts):
            calls.append(("q", texts))
            return super().embed_queries(texts)

    index = KnowledgeIndex(store, Spy())
    index.add_document(1, 10, "規章.pdf", [ParsedChunk(1, "apple banana")])
    index.search(1, "apple", top_k=1)
    assert calls == [("規章.pdf", ["apple banana"]), ("q", ["apple"])]


def test_embedder_failure_propagates(store):
    class Down(FakeEmbedder):
        def embed_queries(self, texts):
            raise KnowledgeStoreError("無法連線到 embedding 服務")

        def embed_documents(self, title, texts):
            raise KnowledgeStoreError("無法連線到 embedding 服務")

    index = KnowledgeIndex(store, Down())
    with pytest.raises(KnowledgeStoreError, match="無法連線到 embedding 服務"):
        index.search(1, "x", top_k=1)
    with pytest.raises(KnowledgeStoreError, match="無法連線到 embedding 服務"):
        index.add_document(1, 10, "a.pdf", [ParsedChunk(1, "x")])


# ---- page_image chunks use the image + text embedding ----


class ImageSpy(FakeEmbedder):
    def __init__(self):
        self.calls = []

    def embed_documents(self, title, texts):
        self.calls.append(("documents", title, list(texts)))
        return super().embed_documents(title, texts)

    def embed_image_texts(self, items):
        self.calls.append(("image_texts", list(items)))
        return super().embed_image_texts(items)


def test_page_image_chunks_embed_png_bytes_with_text(store, tmp_path):
    (tmp_path / "page_1.png").write_bytes(b"png-one")
    spy = ImageSpy()
    index = KnowledgeIndex(store, spy)
    chunks = [
        ParsedChunk(1, "page one", page_image=ImageRef("page_1.png", 10, 10), kind="page_image"),
        ParsedChunk(2, "plain text page"),
    ]
    assert index.add_document(1, 10, "a.pdf", chunks, image_dir=tmp_path) == 2
    assert ("image_texts", [(b"png-one", "page one")]) in spy.calls
    assert ("documents", "a.pdf", ["plain text page"]) in spy.calls
    rows = sorted(store.rows.values(), key=lambda r: r.chunk_index)
    assert [(r.page, r.kind) for r in rows] == [(1, "page_image"), (2, "text")]
    assert rows[0].page_image == ImageRef("page_1.png", 10, 10)


def test_page_image_embeddings_are_sent_in_batches_of_eight(store, tmp_path):
    chunks = []
    for n in range(1, 21):
        (tmp_path / f"page_{n}.png").write_bytes(f"png-{n}".encode())
        chunks.append(
            ParsedChunk(n, f"page {n}", page_image=ImageRef(f"page_{n}.png", 1, 1), kind="page_image")
        )
    spy = ImageSpy()
    index = KnowledgeIndex(store, spy)
    assert index.add_document(1, 10, "a.pdf", chunks, image_dir=tmp_path) == 20
    sizes = [len(c[1]) for c in spy.calls if c[0] == "image_texts"]
    assert sizes == [8, 8, 4]
    # vectors line up with their pages
    rows = sorted(store.rows.values(), key=lambda r: r.chunk_index)
    assert [r.page for r in rows] == list(range(1, 21))
    assert rows[19].dense == FakeEmbedder().embed_image_texts([(b"png-20", "page 20")])[0]


def test_page_image_without_image_dir_is_an_error(store):
    index = KnowledgeIndex(store, FakeEmbedder())
    chunk = ParsedChunk(1, "x", page_image=ImageRef("page_1.png", 1, 1), kind="page_image")
    with pytest.raises(ValueError):
        index.add_document(1, 10, "a.pdf", [chunk])


def test_page_image_chunk_with_blank_text_is_kept(store, tmp_path):
    (tmp_path / "page_1.png").write_bytes(b"png")
    index = KnowledgeIndex(store, FakeEmbedder())
    chunk = ParsedChunk(1, "", page_image=ImageRef("page_1.png", 1, 1), kind="page_image")
    assert index.add_document(1, 10, "a.pdf", [chunk], image_dir=tmp_path) == 1


def test_add_document_without_chunks_clears_old_ones(index, store):
    index.add_document(1, 10, "a.pdf", [ParsedChunk(1, "apple")])
    assert index.add_document(1, 10, "a.pdf", [ParsedChunk(1, "  ")]) == 0
    assert store.rows == {}


def test_liteparse_embeds_pages_in_batches(store, tmp_path):
    """A long scanned PDF: each page image goes to the embedder in batches of at most 8,
    together with that page's LiteParse text, never in one request."""
    from app.kb.parsers.liteparse_parser import LiteParseParser
    from tests.pdf_factory import make_pdf

    path = tmp_path / "long.pdf"
    path.write_bytes(make_pdf([f"page number {n}" for n in range(1, 11)]))
    out = tmp_path / "doc"
    out.mkdir()
    chunks = LiteParseParser("eng").parse(str(path), out, lambda d, t: None)
    spy = ImageSpy()
    index = KnowledgeIndex(store, spy)
    assert index.add_document(1, 10, "long.pdf", chunks, image_dir=out) == 10
    batches = [c[1] for c in spy.calls if c[0] == "image_texts"]
    assert [len(b) for b in batches] == [8, 2]
    assert not any(c[0] == "documents" for c in spy.calls)
    sent = [item for batch in batches for item in batch]
    assert [text for _, text in sent] == [f"page number {n}" for n in range(1, 11)]
    assert sent[0][0] == (out / "page_1.png").read_bytes()
    assert sent[0][0].startswith(b"\x89PNG")
    rows = sorted(store.rows.values(), key=lambda r: r.chunk_index)
    assert [(r.page, r.kind, r.page_image.name) for r in rows] == [
        (n, "page_image", f"page_{n}.png") for n in range(1, 11)
    ]

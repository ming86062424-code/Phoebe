import pytest

from app.kb.index import KnowledgeIndex
from app.kb.parsers.base import PageText
from tests.fakes import KeywordEmbedding


@pytest.fixture
def index(tmp_path):
    return KnowledgeIndex(str(tmp_path / "chroma"), KeywordEmbedding())


def test_search_returns_best_page_with_metadata(index):
    pages = [PageText(1, "apple banana orchard"), PageText(2, "rocket engine fuel")]
    assert index.add_document(1, 10, "a.pdf", pages) == 2
    results = index.search(1, "rocket fuel", top_k=1)
    assert len(results) == 1
    assert (results[0].document_id, results[0].filename, results[0].page) == (10, "a.pdf", 2)
    assert "rocket" in results[0].text


def test_search_empty_kb_returns_nothing(index):
    assert index.search(99, "anything", top_k=3) == []


def test_kbs_are_isolated(index):
    index.add_document(1, 10, "a.pdf", [PageText(1, "apple")])
    index.add_document(2, 20, "b.pdf", [PageText(1, "apple")])
    assert {r.document_id for r in index.search(1, "apple", top_k=5)} == {10}


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


def test_search_is_not_blocked_while_another_document_is_embedding(tmp_path):
    import threading

    started, release = threading.Event(), threading.Event()

    class SlowTextEmbedding(KeywordEmbedding):
        def _get_text_embedding(self, text):
            if "slow" in text:
                started.set()
                release.wait(10)
            return super()._get_text_embedding(text)

        def _get_text_embeddings(self, texts):
            return [self._get_text_embedding(t) for t in texts]

    index = KnowledgeIndex(str(tmp_path / "chroma"), SlowTextEmbedding())
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


def test_chroma_host_uses_http_client(monkeypatch, tmp_path):
    import chromadb

    calls = {}

    def fake_http_client(**kwargs):
        calls.update(kwargs)
        return object()

    monkeypatch.setattr(chromadb, "HttpClient", fake_http_client)
    KnowledgeIndex(str(tmp_path / "chroma"), KeywordEmbedding(), chroma_host="chroma", chroma_port=9000)
    assert (calls["host"], calls["port"]) == ("chroma", 9000)
    assert calls["settings"].anonymized_telemetry is False

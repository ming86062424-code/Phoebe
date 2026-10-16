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

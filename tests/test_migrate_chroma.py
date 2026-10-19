import pytest

chromadb = pytest.importorskip("chromadb")
pytest.importorskip("llama_index.vector_stores.chroma")

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.db import connect, init_db
from app.kb import store as kb_store
from app.kb.index import KnowledgeIndex
from app.kb.migrate_chroma import migrate
from tests.fakes import KeywordEmbedding
from tests.fakes_store import InMemoryChunkStore


@pytest.fixture
def conn(tmp_path):
    c = connect(str(tmp_path / "m.db"))
    init_db(c)
    yield c
    c.close()


@pytest.fixture
def chroma(tmp_path):
    return chromadb.PersistentClient(
        path=str(tmp_path / "chroma"), settings=chromadb.Settings(anonymized_telemetry=False)
    )


@pytest.fixture
def chunk_store():
    return InMemoryChunkStore()


@pytest.fixture
def index(chunk_store):
    return KnowledgeIndex(chunk_store, KeywordEmbedding())


def make_kb(conn):
    user = conn.execute("SELECT id FROM users LIMIT 1").fetchone()
    if user is None:
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES ('u', 'x', 'now')"
        )
        conn.commit()
        user = conn.execute("SELECT id FROM users LIMIT 1").fetchone()
    return kb_store.create_kb(conn, user["id"], "kb", "")["id"]


def make_doc(conn, kb_id, filename="a.pdf"):
    return kb_store.create_document(conn, kb_id, filename, "fake", 1, status="ready")["id"]


def write_old(chroma, kb_id, doc_id, filename, chunks):
    """Write chunks exactly as the old KnowledgeIndex did: LlamaIndex's ChromaVectorStore."""
    collection = chroma.get_or_create_collection(f"kb_{kb_id}", metadata={"hnsw:space": "cosine"})
    embed = KeywordEmbedding()
    nodes = []
    for page, text in chunks:
        node = TextNode(
            text=text,
            metadata={"kb_document_id": doc_id, "filename": filename, "page": page},
            excluded_embed_metadata_keys=["kb_document_id", "page"],
            excluded_llm_metadata_keys=["kb_document_id"],
        )
        node.embedding = embed.get_text_embedding(text)
        nodes.append(node)
    VectorStoreIndex.from_vector_store(
        ChromaVectorStore(chroma_collection=collection), embed_model=embed
    ).insert_nodes(nodes)


def stored(chunk_store, doc_id):
    rows = [r for r in chunk_store.rows.values() if r.document_id == doc_id]
    return sorted(rows, key=lambda r: r.chunk_index)


def test_migrates_chunks_with_order_and_metadata(conn, chroma, index, chunk_store):
    kb = make_kb(conn)
    doc = make_doc(conn, kb, "規章.pdf")
    old = [(1, "第一段 內容"), (1, "第二段 內容"), (2, "第三段 編號 2026-1018")]
    write_old(chroma, kb, doc, "規章.pdf", old)

    result = migrate(chroma, conn, index, dry_run=False)

    assert result == {kb: (3, 0)}
    rows = stored(chunk_store, doc)
    assert [(r.page, r.text) for r in rows] == old
    assert {(r.kb_id, r.filename) for r in rows} == {(kb, "規章.pdf")}
    assert len(rows[0].dense) == 64


def test_skips_deleted_documents_and_kbs(conn, chroma, index, chunk_store):
    kb = make_kb(conn)
    live = make_doc(conn, kb)
    write_old(chroma, kb, live, "a.pdf", [(1, "alive")])
    write_old(chroma, kb, 999, "gone.pdf", [(1, "orphan one"), (2, "orphan two")])
    write_old(chroma, 777, 5, "x.pdf", [(1, "dead kb")])

    result = migrate(chroma, conn, index, dry_run=False)

    assert result[kb] == (1, 2)
    assert result[777] == (0, 1)
    assert {r.document_id for r in chunk_store.rows.values()} == {live}


def test_migration_is_idempotent(conn, chroma, index, chunk_store):
    kb = make_kb(conn)
    doc = make_doc(conn, kb)
    write_old(chroma, kb, doc, "a.pdf", [(1, "alpha"), (2, "beta")])

    migrate(chroma, conn, index, dry_run=False)
    first = {k: r.text for k, r in chunk_store.rows.items()}
    migrate(chroma, conn, index, dry_run=False)

    assert len(chunk_store.rows) == 2
    assert {k: r.text for k, r in chunk_store.rows.items()} == first


def test_dry_run_writes_nothing(conn, chroma, index, chunk_store):
    kb = make_kb(conn)
    doc = make_doc(conn, kb)
    write_old(chroma, kb, doc, "a.pdf", [(1, "alpha"), (2, "beta")])

    result = migrate(chroma, conn, index, dry_run=True)

    assert result == {kb: (2, 0)}
    assert chunk_store.rows == {}
    assert chunk_store.dim is None


def test_ignores_unrelated_collections(conn, chroma, index):
    chroma.get_or_create_collection("something_else")
    assert migrate(chroma, conn, index, dry_run=False) == {}

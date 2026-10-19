from pymilvus import DataType, Function, FunctionType, MilvusClient
from pymilvus.exceptions import ConnectError, MilvusException, MilvusUnavailableException

from app.kb.chunk_store import ChunkRecord, Hit, KnowledgeStoreError

BATCH_SIZE = 256
CONNECT_ERROR = "無法連線到 Milvus"
OUTPUT_FIELDS = ["kb_id", "document_id", "filename", "page", "text"]


class MilvusChunkStore:
    """Chunks in one Milvus collection: a dense vector plus a BM25 sparse vector per chunk."""

    def __init__(self, uri: str, token: str, collection: str, client=None):
        self._uri = uri
        self._token = token
        self._collection = collection
        self._client = client

    def _conn(self):
        if self._client is None:
            try:
                self._client = MilvusClient(uri=self._uri, token=self._token)
            except Exception as exc:
                raise _translate(exc) from exc
        return self._client

    def _exists(self) -> bool:
        try:
            return bool(self._conn().has_collection(self._collection))
        except Exception as exc:
            raise _translate(exc) from exc

    def ensure_ready(self, dim: int) -> None:
        try:
            client = self._conn()
            if client.has_collection(self._collection):
                existing = _dense_dim(client.describe_collection(self._collection))
                if existing != dim:
                    raise KnowledgeStoreError(
                        f"EMBED_MODEL 的向量維度（{dim}）與既有知識庫（{existing}）不同，請重建知識庫"
                    )
            else:
                self._create(client, dim)
        except Exception as exc:
            raise _translate(exc) from exc

    def _create(self, client, dim: int) -> None:
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("kb_id", DataType.INT64, is_partition_key=True)
        schema.add_field("document_id", DataType.INT64)
        schema.add_field("page", DataType.INT64)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("filename", DataType.VARCHAR, max_length=512)
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=8192,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        )
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="text_bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse"],
            )
        )
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(
            field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25"
        )
        client.create_collection(
            self._collection, schema=schema, index_params=index_params, consistency_level="Strong"
        )

    def insert(self, records: list[ChunkRecord]) -> None:
        rows = [
            {
                "id": f"{r.document_id}:{r.chunk_index}",
                "kb_id": r.kb_id,
                "document_id": r.document_id,
                "page": r.page,
                "chunk_index": r.chunk_index,
                "filename": r.filename,
                "text": r.text,
                "dense": r.dense,
            }
            for r in records
        ]
        try:
            client = self._conn()
            for start in range(0, len(rows), BATCH_SIZE):
                client.insert(self._collection, rows[start : start + BATCH_SIZE])
        except Exception as exc:
            raise _translate(exc) from exc

    def delete_document(self, kb_id: int, document_id: int) -> None:
        self._delete(f"kb_id == {int(kb_id)} and document_id == {int(document_id)}")

    def delete_kb(self, kb_id: int) -> None:
        self._delete(f"kb_id == {int(kb_id)}")

    def _delete(self, expr: str) -> None:
        if not self._exists():
            return  # nothing was ever indexed
        try:
            self._conn().delete(self._collection, filter=expr)
        except Exception as exc:
            raise _translate(exc) from exc

    def dense_search(self, kb_id: int, vector: list[float], limit: int) -> list[Hit]:
        return self._search(kb_id, [vector], "dense", {"metric_type": "COSINE"}, limit)

    def keyword_search(self, kb_id: int, query: str, limit: int) -> list[Hit]:
        # BM25 takes the raw text as the search data; it never touches the filter string.
        if not query.strip():
            return []
        return self._search(kb_id, [query], "sparse", {"metric_type": "BM25"}, limit)

    def _search(self, kb_id, data, field, params, limit) -> list[Hit]:
        if not self._exists():
            return []
        try:
            results = self._conn().search(
                self._collection,
                data=data,
                anns_field=field,
                filter=f"kb_id == {int(kb_id)}",
                search_params=params,
                limit=limit,
                output_fields=OUTPUT_FIELDS,
                consistency_level="Strong",
            )
        except Exception as exc:
            raise _translate(exc) from exc
        return [
            Hit(
                chunk_id=str(h["id"]),
                kb_id=int(h["entity"]["kb_id"]),
                document_id=int(h["entity"]["document_id"]),
                filename=h["entity"]["filename"],
                page=int(h["entity"]["page"]),
                text=h["entity"]["text"],
                score=float(h["distance"]),
            )
            for h in (results[0] if results else [])
        ]


def _dense_dim(description: dict) -> int:
    for f in description["fields"]:
        if f["name"] == "dense":
            return int(f["params"]["dim"])
    raise KnowledgeStoreError("既有的 Milvus collection 缺少 dense 欄位，請重建知識庫")


def _translate(exc: Exception) -> KnowledgeStoreError:
    if isinstance(exc, KnowledgeStoreError):
        return exc
    if isinstance(exc, (ConnectError, MilvusUnavailableException, ConnectionError, OSError)):
        return KnowledgeStoreError(CONNECT_ERROR)
    if isinstance(exc, MilvusException):
        # pymilvus raises a plain MilvusException when the first connection fails
        if "Fail connecting" in exc.message or "server unavailable" in exc.message:
            return KnowledgeStoreError(CONNECT_ERROR)
        return KnowledgeStoreError(f"Milvus 錯誤：{exc.message}")
    return KnowledgeStoreError(f"Milvus 錯誤：{exc}")

import json
import logging

import grpc
from pymilvus import DataType, Function, FunctionType, MilvusClient
from pymilvus.client.types import LoadState, Status
from pymilvus.exceptions import ConnectError, MilvusException, MilvusUnavailableException

from app.kb.chunk_store import ChunkRecord, Hit, KnowledgeStoreError
from app.kb.parsers.base import ImageRef

logger = logging.getLogger(__name__)

BATCH_SIZE = 256
TIMEOUT = 10  # seconds; also bounds pymilvus's retry loop (75 retries otherwise)
TEXT_MAX_BYTES = 65535  # VARCHAR length counts UTF-8 bytes; 8192 chars x 4 bytes fits
FILENAME_MAX_BYTES = 2048
KIND_MAX_BYTES = 16  # "text" or "page_image"
# JSON [{"name","w","h"}]; one entry is ~45 bytes, so ~180 images per page fit
IMAGES_MAX_BYTES = 8192
# JSON {"name","w","h"} of a "page_{n}.png"; empty string when the chunk has none
PAGE_IMAGE_MAX_BYTES = 256
CONNECT_ERROR = "無法連線到 Milvus"
CREATE_TIMEOUT = 60  # creating a collection with indexes can take longer than a normal call
ANALYZER_CHANGED = "知識庫的斷詞設定已更新，請重建知識庫（刪除 collection 後重新遷移或重新上傳）"
SCHEMA_CHANGED = "知識庫結構已更新，請重建知識庫（刪除 collection 後重新處理文件）"
REQUIRED_FIELDS = ("kind", "images", "page_image")
# Question words and particles would otherwise dominate BM25 for natural-language queries.
EN_STOP = [
    "_english_", "i", "me", "my", "we", "our", "you", "your", "he", "she", "him", "her", "its",
    "what", "which", "who", "whom", "how", "when", "where", "why", "do", "does", "did", "doing",
    "am", "been", "being", "have", "has", "had", "can", "could", "would", "should", "shall",
    "may", "might", "must", "much", "many", "some", "any", "all", "about", "from", "up", "so",
    "than", "too", "very", "just", "s", "t", "d", "ll", "re", "ve", "m",
]  # fmt: skip
ZH_STOP = [
    "的", "了", "是", "在", "和", "與", "及", "或", "也", "就", "都", "而", "被", "把", "我", "你",
    "他", "她", "它", "我們", "請", "嗎", "呢", "吧", "啊", "這", "那", "之", "於", "以",
]  # fmt: skip
TEXT_ANALYZER = {
    "tokenizer": "jieba",
    "filter": ["cnalphanumonly", "lowercase", {"type": "stop", "stop_words": EN_STOP + ZH_STOP}],
}
OUTPUT_FIELDS = [
    "kb_id", "document_id", "filename", "page", "text", "kind", "images", "page_image",
]  # fmt: skip


class MilvusChunkStore:
    """Chunks in one Milvus collection: a dense vector plus a BM25 sparse vector per chunk."""

    def __init__(self, uri: str, token: str, collection: str, client=None):
        self._uri = uri
        self._token = token
        self._collection = collection
        self._client = client
        self._exists_cached = False

    def _conn(self):
        if self._client is None:
            try:
                self._client = MilvusClient(uri=self._uri, token=self._token, timeout=TIMEOUT)
            except Exception as exc:
                raise _translate(exc) from exc
        return self._client

    def _exists(self) -> bool:
        if self._exists_cached:
            return True
        try:
            found = bool(self._conn().has_collection(self._collection, timeout=TIMEOUT))
        except Exception as exc:
            raise _translate(exc) from exc
        self._exists_cached = found
        return found

    def ensure_ready(self, dim: int) -> None:
        try:
            client = self._conn()
            if not client.has_collection(self._collection, timeout=TIMEOUT):
                try:
                    self._create(client, dim)
                    self._exists_cached = True
                    return
                except Exception:
                    # web and worker may start together: the other one created it first
                    if not client.has_collection(self._collection, timeout=TIMEOUT):
                        raise
            description = client.describe_collection(self._collection, timeout=TIMEOUT)
            names = {f["name"] for f in description["fields"]}
            if any(name not in names for name in REQUIRED_FIELDS):
                raise KnowledgeStoreError(SCHEMA_CHANGED)
            existing = _dense_dim(description)
            if existing != dim:
                raise KnowledgeStoreError(
                    f"embedding 模型的向量維度（{dim}）與既有知識庫（{existing}）不同，請重建知識庫"
                )
            if _text_analyzer(description) != TEXT_ANALYZER:
                raise KnowledgeStoreError(ANALYZER_CHANGED)
            self._ensure_loaded(client)
            self._exists_cached = True
        except Exception as exc:
            raise _translate(exc) from exc

    def _ensure_loaded(self, client) -> None:
        state = client.get_load_state(self._collection, timeout=TIMEOUT).get("state")
        if state != LoadState.Loaded:
            client.load_collection(self._collection, timeout=CREATE_TIMEOUT)

    def _create(self, client, dim: int) -> None:
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("kb_id", DataType.INT64, is_partition_key=True)
        schema.add_field("document_id", DataType.INT64)
        schema.add_field("page", DataType.INT64)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("filename", DataType.VARCHAR, max_length=FILENAME_MAX_BYTES)
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=TEXT_MAX_BYTES,
            enable_analyzer=True,
            analyzer_params=TEXT_ANALYZER,
        )
        schema.add_field("kind", DataType.VARCHAR, max_length=KIND_MAX_BYTES)
        schema.add_field("images", DataType.VARCHAR, max_length=IMAGES_MAX_BYTES)
        schema.add_field("page_image", DataType.VARCHAR, max_length=PAGE_IMAGE_MAX_BYTES)
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
            self._collection,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
            timeout=CREATE_TIMEOUT,
        )

    def insert(self, records: list[ChunkRecord]) -> None:
        rows = [
            {
                "id": f"{r.document_id}:{r.chunk_index}",
                "kb_id": r.kb_id,
                "document_id": r.document_id,
                "page": r.page,
                "chunk_index": r.chunk_index,
                "filename": _truncate_bytes(r.filename, FILENAME_MAX_BYTES),
                "text": _truncate_bytes(r.text, TEXT_MAX_BYTES),
                "kind": r.kind,
                "images": _images_json(r.images),
                "page_image": _page_image_json(r.page_image),
                "dense": r.dense,
            }
            for r in records
        ]
        try:
            client = self._conn()
            for start in range(0, len(rows), BATCH_SIZE):
                client.insert(self._collection, rows[start : start + BATCH_SIZE], timeout=TIMEOUT)
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
            self._conn().delete(self._collection, filter=expr, timeout=TIMEOUT)
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
                timeout=TIMEOUT,
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
                kind=h["entity"].get("kind") or "text",
                images=_parse_images(h["entity"].get("images")),
                page_image=_parse_ref(h["entity"].get("page_image")),
            )
            for h in (results[0] if results else [])
        ]


def _ref_json(ref: ImageRef) -> str:
    return json.dumps({"name": ref.name, "w": ref.width, "h": ref.height}, ensure_ascii=False)


def _page_image_json(ref: ImageRef | None) -> str:
    """The page image reference; ImageRef bounds the name and size, so it always fits."""
    if ref is None:
        return ""
    value = _ref_json(ref)
    if len(value.encode("utf-8")) > PAGE_IMAGE_MAX_BYTES:  # never cut JSON in half
        raise ValueError(f"page image reference too long: {ref.name!r}")
    return value


def _images_json(images: list[ImageRef]) -> str:
    """JSON array of the images; trailing ones that don't fit the field are dropped."""
    entries: list[str] = []
    size = 2  # the brackets
    for ref in images:
        entry = _ref_json(ref)
        extra = len(entry.encode("utf-8")) + (1 if entries else 0)
        if size + extra > IMAGES_MAX_BYTES:
            logger.warning("chunk has %d images; only %d fit", len(images), len(entries))
            break
        entries.append(entry)
        size += extra
    return "[" + ",".join(entries) + "]"


def _to_ref(item) -> ImageRef | None:
    try:
        return ImageRef(str(item["name"]), int(item["w"]), int(item["h"]))
    except (KeyError, TypeError, ValueError):
        return None


def _parse_ref(raw: str | None) -> ImageRef | None:
    if not raw:
        return None
    try:
        return _to_ref(json.loads(raw))
    except ValueError:
        return None


def _parse_images(raw: str | None) -> list[ImageRef]:
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except ValueError:
        return []
    if not isinstance(items, list):
        return []
    return [ref for ref in map(_to_ref, items) if ref is not None]


def _text_analyzer(description: dict) -> dict | None:
    for f in description["fields"]:
        if f["name"] == "text":
            raw = f.get("params", {}).get("analyzer_params")
            return json.loads(raw) if isinstance(raw, str) else raw
    return None


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
        # first connect fails with Status.CONNECT_FAILED; later RPCs end with the gRPC
        # code once pymilvus's retries are used up or time out
        if exc.code in (Status.CONNECT_FAILED, grpc.StatusCode.UNAVAILABLE):
            return KnowledgeStoreError(CONNECT_ERROR)
        if "Fail connecting" in exc.message:
            return KnowledgeStoreError(CONNECT_ERROR)
    if isinstance(exc, MilvusException) and exc.code == grpc.StatusCode.DEADLINE_EXCEEDED:
        return KnowledgeStoreError("Milvus 回應逾時")
    logger.error("Milvus request failed: %r", exc)
    return KnowledgeStoreError("Milvus 錯誤")


def _truncate_bytes(value: str, limit: int) -> str:
    """Cut to at most `limit` UTF-8 bytes without splitting a character."""
    data = value.encode("utf-8")
    if len(data) <= limit:
        return value
    return data[:limit].decode("utf-8", errors="ignore")

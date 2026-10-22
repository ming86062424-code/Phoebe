import logging
import sqlite3
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from app.auth import current_user
from app.config import Settings
from app.deps import get_db, get_kb_services, get_settings
from app.kb import store
from app.kb.filetypes import IMAGE_EXTENSIONS, PDF_EXTENSIONS, TEXT_EXTENSIONS, UPLOAD_EXTENSIONS
from app.kb.parsers.base import IMAGE_NAME
from app.kb.parsers.images import open_image
from app.kb.parsers.text_parser import decode_text
from app.kb.services import KbServices

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

MAX_NAME_LENGTH = 50
MAX_DESCRIPTION_LENGTH = 500
UPLOAD_CHUNK = 1024 * 1024
# room for the multipart boundaries and the parser field around the file itself
MULTIPART_OVERHEAD = 64 * 1024
MAX_FILENAME_LENGTH = 200
UNSUPPORTED_TYPE = "不支援的檔案類型（支援 PDF、txt、md、png、jpg、jpeg、webp）"
UNREADABLE = "無法讀取檔案內容"


def _short_filename(filename: str) -> str:
    """At most 200 characters, keeping the extension (it decides the stored file's type)."""
    if len(filename) <= MAX_FILENAME_LENGTH:
        return filename
    suffix = Path(filename).suffix
    return filename[: MAX_FILENAME_LENGTH - len(suffix)] + suffix


def _readable(path: Path, suffix: str) -> bool:
    """Whether the content matches its extension: PDF magic, an image of the extension's
    format that fully decodes within the pixel cap, or UTF-8 (with or without BOM) text."""
    try:
        if suffix in PDF_EXTENSIONS:
            with path.open("rb") as f:
                return f.read(5) == b"%PDF-"
        if suffix in IMAGE_EXTENSIONS:
            open_image(path, suffix)
            return True
        if suffix in TEXT_EXTENSIONS:
            decode_text(path.read_bytes())
            return True
    except Exception:  # PIL raises many kinds (including decompression bombs)
        return False
    return False


def _clean_name(v: str) -> str:
    v = v.strip()
    if not v:
        raise ValueError("名稱不能是空白")
    if len(v) > MAX_NAME_LENGTH:
        raise ValueError(f"名稱最多 {MAX_NAME_LENGTH} 字")
    return v


def _clean_description(v: str) -> str:
    if len(v) > MAX_DESCRIPTION_LENGTH:
        raise ValueError(f"說明最多 {MAX_DESCRIPTION_LENGTH} 字")
    return v.strip()


class KbIn(BaseModel):
    name: str
    description: str = ""
    parser: str = "liteparse"  # for the PDFs and images uploaded to it

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return _clean_name(v)

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        return _clean_description(v)


class KbPatch(BaseModel):
    name: str | None = None
    description: str | None = None
    parser: str | None = None  # only documents uploaded afterwards use it
    # hybrid search weight of the vector path; null follows KB_VECTOR_WEIGHT again
    vector_weight: float | None = Field(default=None, ge=0, le=1)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
        return None if v is None else _clean_name(v)

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str | None) -> str | None:
        return None if v is None else _clean_description(v)


def _check_parser(kb: KbServices, parser: str | None) -> None:
    """A knowledge base's parser must be one users can pick for PDFs and images (not the
    text parser). It may be unavailable right now: uploads say so until it is back."""
    if parser is not None and parser not in kb.parsers:
        raise HTTPException(status_code=422, detail="未知的解析器")


def _require_kb(conn: sqlite3.Connection, user: dict, kb_id: int) -> dict:
    kb = store.get_kb(conn, user["id"], kb_id)
    if kb is None:
        raise HTTPException(status_code=404, detail="找不到知識庫")
    return kb


@router.get("/parsers")
async def list_parsers(
    _: dict = Depends(current_user), kb: KbServices = Depends(get_kb_services)
):
    return await run_in_threadpool(kb.parser_info)


@router.get("/kbs")
def list_kbs(user: dict = Depends(current_user), conn: sqlite3.Connection = Depends(get_db)):
    return store.list_kbs(conn, user["id"])


@router.post("/kbs", status_code=201)
def create_kb(
    body: KbIn,
    user: dict = Depends(current_user),
    conn: sqlite3.Connection = Depends(get_db),
    kb: KbServices = Depends(get_kb_services),
):
    _check_parser(kb, body.parser)
    return store.create_kb(conn, user["id"], body.name, body.description, body.parser)


@router.patch("/kbs/{kb_id}")
def update_kb(
    kb_id: int,
    body: KbPatch,
    user: dict = Depends(current_user),
    conn: sqlite3.Connection = Depends(get_db),
    services: KbServices = Depends(get_kb_services),
):
    _check_parser(services, body.parser)
    kb = store.update_kb(
        conn,
        user["id"],
        kb_id,
        name=body.name,
        description=body.description,
        parser=body.parser,
        vector_weight=(
            body.vector_weight if "vector_weight" in body.model_fields_set else store.UNCHANGED
        ),
    )
    if kb is None:
        raise HTTPException(status_code=404, detail="找不到知識庫")
    return kb


@router.delete("/kbs/{kb_id}", status_code=204)
def delete_kb(
    kb_id: int,
    user: dict = Depends(current_user),
    conn: sqlite3.Connection = Depends(get_db),
    kb: KbServices = Depends(get_kb_services),
):
    _require_kb(conn, user, kb_id)
    store.delete_kb(conn, user["id"], kb_id)
    # the worker notices the deleted rows and stops; files still in use are removed later
    try:
        kb.index.delete_kb(kb_id)
    except Exception as e:
        # ids are never reused, so leftover chunks are never searched
        logger.warning("刪除知識庫 %s 的向量失敗：%s", kb_id, e, exc_info=True)
    kb.worker.remove_path(kb.kb_dir / str(kb_id))
    return Response(status_code=204)


@router.get("/kbs/{kb_id}/documents")
def list_documents(
    kb_id: int, user: dict = Depends(current_user), conn: sqlite3.Connection = Depends(get_db)
):
    _require_kb(conn, user, kb_id)
    return store.list_documents(conn, kb_id)


@router.post("/kbs/{kb_id}/documents", status_code=201)
async def upload_document(
    kb_id: int,
    request: Request,
    user: dict = Depends(current_user),
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
    kb: KbServices = Depends(get_kb_services),
):
    """multipart form: file.

    PDFs and images use the knowledge base's parser setting; .txt and .md always use the
    text parser. A `parser` form field from older clients is ignored.
    """
    kb_row = _require_kb(conn, user, kb_id)
    limit = settings.max_upload_mb * 1024 * 1024
    too_big = HTTPException(status_code=413, detail=f"檔案超過 {settings.max_upload_mb}MB 上限")
    # The form is parsed here rather than by FastAPI so an oversized upload is
    # refused before its body is read.
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit + MULTIPART_OVERHEAD:
        raise too_big
    form = await request.form()
    file = form.get("file")
    if not isinstance(file, UploadFile):
        raise HTTPException(status_code=422, detail="請選擇要上傳的檔案")
    filename = _short_filename((file.filename or "").strip())
    suffix = Path(filename).suffix.lower()
    if suffix not in UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=422, detail=UNSUPPORTED_TYPE)
    if suffix in TEXT_EXTENSIONS:
        chosen = kb.text_parser
    else:
        chosen = kb.parsers.get(kb_row["parser"])
        if chosen is None:  # e.g. the parser was removed from this installation
            raise HTTPException(
                status_code=422,
                detail=f"知識庫設定的解析器 {kb_row['parser']} 已不存在，請到知識庫設定更換",
            )
        available, reason = await run_in_threadpool(kb.availability, chosen.id)
        if not available:
            raise HTTPException(status_code=422, detail=reason or "此解析器目前無法使用")

    def save() -> dict:
        # blocking file and database work, run in a worker thread
        folder = kb.kb_dir / str(kb_id)
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f"upload-{uuid.uuid4().hex}.tmp"
        size = 0
        try:
            file.file.seek(0)
            with tmp.open("wb") as out:
                while chunk := file.file.read(UPLOAD_CHUNK):
                    size += len(chunk)
                    if size > limit:
                        raise too_big
                    out.write(chunk)
            if not _readable(tmp, suffix):
                raise HTTPException(status_code=422, detail=UNREADABLE)
            # 'pending' can't be claimed: only queue it once its file is in place
            doc = store.create_document(conn, kb_id, filename, chosen.id, size, "pending")
            try:
                tmp.replace(kb.file_path(kb_id, doc["id"], filename))
                store.update_document(conn, doc["id"], status="queued")
            except BaseException:
                store.delete_document(conn, kb_id, doc["id"])
                raise
        finally:
            tmp.unlink(missing_ok=True)
        kb.worker.enqueue(doc["id"])
        return store.get_document(conn, doc["id"])

    try:
        return await run_in_threadpool(save)
    finally:
        await form.close()


@router.delete("/kbs/{kb_id}/documents/{doc_id}", status_code=204)
def delete_document(
    kb_id: int,
    doc_id: int,
    user: dict = Depends(current_user),
    conn: sqlite3.Connection = Depends(get_db),
    kb: KbServices = Depends(get_kb_services),
):
    _require_kb(conn, user, kb_id)
    doc = store.get_document(conn, doc_id)
    if doc is None or doc["kb_id"] != kb_id or not store.delete_document(conn, kb_id, doc_id):
        raise HTTPException(status_code=404, detail="找不到文件")
    # the worker notices the deleted row and stops; a file still in use is removed later
    try:
        kb.index.delete_document(kb_id, doc_id)
    except Exception as e:
        logger.warning("刪除文件 %s 的向量失敗：%s", doc_id, e, exc_info=True)
    kb.worker.remove_path(kb.file_path(kb_id, doc_id, doc["filename"]))
    kb.worker.remove_path(kb.doc_dir(kb_id, doc_id))
    return Response(status_code=204)


@router.get("/kbs/{kb_id}/documents/{doc_id}/chunks")
async def document_chunks(
    kb_id: int,
    doc_id: int,
    user: dict = Depends(current_user),
    conn: sqlite3.Connection = Depends(get_db),
    kb: KbServices = Depends(get_kb_services),
):
    """How the document was split: its stored chunks, in order (empty until indexed)."""
    _require_kb(conn, user, kb_id)
    doc = store.get_document(conn, doc_id)
    if doc is None or doc["kb_id"] != kb_id:
        raise HTTPException(status_code=404, detail="找不到文件")
    try:
        chunks = await run_in_threadpool(kb.index.list_chunks, kb_id, doc_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"無法讀取切塊：{e}") from e
    return {"document": doc, "chunks": [c.to_dict() for c in chunks]}


@router.post("/kbs/{kb_id}/documents/{doc_id}/retry")
def retry_document(
    kb_id: int,
    doc_id: int,
    user: dict = Depends(current_user),
    conn: sqlite3.Connection = Depends(get_db),
    kb: KbServices = Depends(get_kb_services),
):
    _require_kb(conn, user, kb_id)
    doc = store.get_document(conn, doc_id)
    if doc is None or doc["kb_id"] != kb_id:
        raise HTTPException(status_code=404, detail="找不到文件")
    if doc["status"] != "failed":
        raise HTTPException(status_code=409, detail="只有失敗的文件可以重試")
    store.update_document(conn, doc_id, status="queued", error=None, progress_done=0)
    kb.worker.enqueue(doc_id)
    return store.get_document(conn, doc_id)


# Names are reused when a document is reprocessed, so always revalidate (ETag/Last-Modified).
IMAGE_CACHE_CONTROL = "private, no-cache"


@router.get("/kbs/{kb_id}/documents/{doc_id}/images/{name}")
def document_image(
    kb_id: int,
    doc_id: int,
    name: str,
    user: dict = Depends(current_user),
    conn: sqlite3.Connection = Depends(get_db),
    kb: KbServices = Depends(get_kb_services),
):
    """A page or cropped image a parser wrote for the document (PNG).

    Logged out is 401 like every API. After that, anything that is not the owner's
    existing document and a whitelisted name that resolves inside the document's folder
    is 404, so nothing leaks about other users' knowledge bases or other files.
    """
    not_found = HTTPException(status_code=404, detail="找不到圖片")
    if not IMAGE_NAME.fullmatch(name):
        raise not_found
    if store.get_kb(conn, user["id"], kb_id) is None:
        raise not_found
    doc = store.get_document(conn, doc_id)
    if doc is None or doc["kb_id"] != kb_id:
        raise not_found
    folder = kb.doc_dir(kb_id, doc_id).resolve()
    path = (folder / name).resolve()
    if path.parent != folder or not path.is_file():
        raise not_found
    return FileResponse(
        path, media_type="image/png", headers={"Cache-Control": IMAGE_CACHE_CONTROL}
    )


@router.get("/search-settings")
def search_settings(
    _user: dict = Depends(current_user), settings: Settings = Depends(get_settings)
):
    """The system defaults: KB_VECTOR_WEIGHT is the weight of knowledge bases that have
    none of their own."""
    return {"vector_weight": settings.kb_vector_weight, "top_k": settings.kb_top_k}


@router.get("/kbs/{kb_id}/search")
async def search(
    kb_id: int,
    q: str = Query(min_length=1, max_length=1000),
    # for trying weights in the search test; the chat uses the knowledge base's weight
    vector_weight: float | None = Query(default=None, ge=0, le=1),
    user: dict = Depends(current_user),
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
    kb: KbServices = Depends(get_kb_services),
):
    kb_row = _require_kb(conn, user, kb_id)
    weight = (
        store.vector_weight(kb_row, settings.kb_vector_weight)
        if vector_weight is None
        else vector_weight
    )
    try:
        sources = await run_in_threadpool(
            kb.index.search, kb_id, q, settings.kb_top_k, weight
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"知識庫檢索失敗：{e}") from e
    # chunks of a document deleted during a store outage may linger; SQLite is the truth
    live = {d["id"] for d in store.list_documents(conn, kb_id)}
    return [s.to_dict() for s in sources if s.document_id in live]

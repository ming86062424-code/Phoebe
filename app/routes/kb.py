import logging
import sqlite3
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from starlette.datastructures import UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, field_validator

from app.auth import current_user
from app.config import Settings
from app.deps import get_db, get_kb_services, get_settings
from app.kb import store
from app.kb.services import KbServices

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

MAX_NAME_LENGTH = 50
MAX_DESCRIPTION_LENGTH = 500
UPLOAD_CHUNK = 1024 * 1024
# room for the multipart boundaries and the parser field around the file itself
MULTIPART_OVERHEAD = 64 * 1024


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

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
        return None if v is None else _clean_name(v)

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str | None) -> str | None:
        return None if v is None else _clean_description(v)


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
    body: KbIn, user: dict = Depends(current_user), conn: sqlite3.Connection = Depends(get_db)
):
    return store.create_kb(conn, user["id"], body.name, body.description)


@router.patch("/kbs/{kb_id}")
def update_kb(
    kb_id: int,
    body: KbPatch,
    user: dict = Depends(current_user),
    conn: sqlite3.Connection = Depends(get_db),
):
    kb = store.update_kb(conn, user["id"], kb_id, name=body.name, description=body.description)
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
    """multipart form: file (PDF) and parser (default liteparse)."""
    _require_kb(conn, user, kb_id)
    limit = settings.max_upload_mb * 1024 * 1024
    too_big = HTTPException(status_code=413, detail=f"檔案超過 {settings.max_upload_mb}MB 上限")
    # The form is parsed here rather than by FastAPI so an oversized upload is
    # refused before its body is read.
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit + MULTIPART_OVERHEAD:
        raise too_big
    form = await request.form()
    file = form.get("file")
    parser = form.get("parser", "liteparse")
    if not isinstance(file, UploadFile):
        raise HTTPException(status_code=422, detail="請選擇要上傳的檔案")
    chosen = kb.parsers.get(parser) if isinstance(parser, str) else None
    if chosen is None:
        raise HTTPException(status_code=422, detail="未知的解析器")
    available, reason = await run_in_threadpool(kb.availability, chosen.id)
    if not available:
        raise HTTPException(status_code=422, detail=reason or "此解析器目前無法使用")
    filename = (file.filename or "").strip()
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="目前只支援 PDF 檔")

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
            with tmp.open("rb") as f:
                if f.read(5) != b"%PDF-":
                    raise HTTPException(status_code=422, detail="檔案不是有效的 PDF")
            # 'pending' can't be claimed: only queue it once its file is in place
            doc = store.create_document(conn, kb_id, filename[:200], chosen.id, size, "pending")
            try:
                tmp.replace(kb.file_path(kb_id, doc["id"]))
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
    if not store.delete_document(conn, kb_id, doc_id):
        raise HTTPException(status_code=404, detail="找不到文件")
    # the worker notices the deleted row and stops; a file still in use is removed later
    try:
        kb.index.delete_document(kb_id, doc_id)
    except Exception as e:
        logger.warning("刪除文件 %s 的向量失敗：%s", doc_id, e, exc_info=True)
    kb.worker.remove_path(kb.file_path(kb_id, doc_id))
    return Response(status_code=204)


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


@router.get("/kbs/{kb_id}/search")
async def search(
    kb_id: int,
    q: str = Query(min_length=1, max_length=1000),
    user: dict = Depends(current_user),
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
    kb: KbServices = Depends(get_kb_services),
):
    _require_kb(conn, user, kb_id)
    try:
        sources = await run_in_threadpool(kb.index.search, kb_id, q, settings.kb_top_k)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"知識庫檢索失敗：{e}") from e
    # chunks of a document deleted during a store outage may linger; SQLite is the truth
    live = {d["id"] for d in store.list_documents(conn, kb_id)}
    return [s.to_dict() for s in sources if s.document_id in live]

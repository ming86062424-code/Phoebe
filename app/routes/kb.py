import sqlite3
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, field_validator

from app.auth import current_user
from app.config import Settings
from app.deps import get_db, get_kb_services, get_settings
from app.kb import store
from app.kb.services import KbServices

router = APIRouter(prefix="/api")

MAX_NAME_LENGTH = 50
MAX_DESCRIPTION_LENGTH = 500
UPLOAD_CHUNK = 1024 * 1024


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
def list_parsers(_: dict = Depends(current_user), kb: KbServices = Depends(get_kb_services)):
    return kb.parser_info()


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
    kb.index.delete_kb(kb_id)
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
    file: UploadFile = File(...),
    parser: str = Form("liteparse"),
    user: dict = Depends(current_user),
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
    kb: KbServices = Depends(get_kb_services),
):
    _require_kb(conn, user, kb_id)
    chosen = kb.parsers.get(parser)
    if chosen is None:
        raise HTTPException(status_code=422, detail="未知的解析器")
    available, reason = chosen.availability()
    if not available:
        raise HTTPException(status_code=422, detail=reason or "此解析器目前無法使用")
    filename = (file.filename or "").strip()
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="目前只支援 PDF 檔")

    folder = kb.kb_dir / str(kb_id)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"upload-{uuid.uuid4().hex}.tmp"
    limit = settings.max_upload_mb * 1024 * 1024
    size = 0
    try:
        with tmp.open("wb") as out:
            while chunk := await file.read(UPLOAD_CHUNK):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(
                        status_code=413, detail=f"檔案超過 {settings.max_upload_mb}MB 上限"
                    )
                out.write(chunk)
        with tmp.open("rb") as f:
            if f.read(5) != b"%PDF-":
                raise HTTPException(status_code=422, detail="檔案不是有效的 PDF")
        doc = store.create_document(conn, kb_id, filename[:200], parser, size)
        tmp.replace(kb.file_path(kb_id, doc["id"]))
    finally:
        tmp.unlink(missing_ok=True)
    kb.worker.enqueue(doc["id"])
    return store.get_document(conn, doc["id"])


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
    kb.index.delete_document(kb_id, doc_id)
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
    return [s.to_dict() for s in sources]

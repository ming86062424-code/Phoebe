import sqlite3
from dataclasses import dataclass, field

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, field_validator

from app import conversations
from app.auth import current_user
from app.chat_stream import claim_reply, ndjson_reply
from app.config import Settings
from app.deps import get_chat_client, get_db, get_kb_services, get_settings
from app.kb import store as kb_store
from app.kb.rag import build_system_prompt, load_source_images
from app.kb.services import KbServices
from app.llm import ChatClient
from app.llm_settings import load_llm_settings

router = APIRouter(prefix="/api")

MAX_CONTENT_LENGTH = 8000


class MessageIn(BaseModel):
    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("訊息不能是空白")
        if len(stripped) > MAX_CONTENT_LENGTH:
            raise ValueError(f"訊息最多 {MAX_CONTENT_LENGTH} 字")
        return v


@dataclass(frozen=True)
class Knowledge:
    prompt: str | None = None  # system text with the references (and image mapping)
    sources: list[dict] | None = None  # the sources event; None without a knowledge base
    images: list[bytes] = field(default_factory=list)  # PNGs for the last user message


async def _with_knowledge(
    conn: sqlite3.Connection,
    user: dict,
    conv: dict,
    history: list[dict],
    settings: Settings,
    kb: KbServices,
) -> Knowledge:
    """Retrieve for the last user message; an empty Knowledge without a knowledge base."""
    if conv["kb_id"] is None:
        return Knowledge()
    kb_id = conv["kb_id"]
    kb_row = kb_store.get_kb(conn, user["id"], kb_id)
    if kb_row is None:
        return Knowledge()
    question = next(m["content"] for m in reversed(history) if m["role"] == "user")
    try:
        found = await run_in_threadpool(
            kb.index.search, kb_id, question, settings.kb_top_k, settings.kb_vector_weight
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"知識庫檢索失敗：{e}") from e
    live = {d["id"] for d in kb_store.list_documents(conn, kb_id)}
    found = [s for s in found if s.document_id in live]
    images = await run_in_threadpool(
        load_source_images,
        found,
        lambda doc_id: kb.doc_dir(kb_id, doc_id),
        settings.kb_max_images,
    )
    return Knowledge(
        build_system_prompt(kb_row["name"], found, images),
        # kb_id lets the UI fetch the source's images after the conversation changes knowledge base
        [{**s.to_dict(), "kb_id": kb_id} for s in found],
        [image.data for image in images],
    )


def _combine_system(system_prompt: str, kb_prompt: str | None) -> str | None:
    return "\n\n".join(p for p in (system_prompt, kb_prompt) if p) or None


@router.post("/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: int,
    body: MessageIn,
    user: dict = Depends(current_user),
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
    client: ChatClient = Depends(get_chat_client),
    kb: KbServices = Depends(get_kb_services),
):
    conv = conversations.get_conversation(conn, user["id"], conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    release = await run_in_threadpool(
        claim_reply, settings.db_path, conversation_id, settings.reply_lock_ttl_seconds
    )
    release.start_keep_alive()
    try:
        conversations.add_message(conn, conversation_id, "user", body.content)
        conversations.set_title_if_empty(conn, conversation_id, body.content.strip()[:30])
        history = conversations.get_history(conn, conversation_id)
        knowledge = await _with_knowledge(conn, user, conv, history, settings, kb)
        llm = await run_in_threadpool(load_llm_settings, conn, settings)
    except BaseException:
        release()
        raise

    return await ndjson_reply(
        client,
        history,
        conversation_id,
        settings.db_path,
        release,
        knowledge.sources,
        system=_combine_system(llm.system_prompt, knowledge.prompt),
        temperature=llm.temperature,
        images=knowledge.images,
    )


@router.post("/conversations/{conversation_id}/regenerate")
async def regenerate(
    conversation_id: int,
    user: dict = Depends(current_user),
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
    client: ChatClient = Depends(get_chat_client),
    kb: KbServices = Depends(get_kb_services),
):
    conv = conversations.get_conversation(conn, user["id"], conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    release = await run_in_threadpool(
        claim_reply, settings.db_path, conversation_id, settings.reply_lock_ttl_seconds
    )
    release.start_keep_alive()
    history = conversations.get_history(conn, conversation_id)
    if not history or history[-1]["role"] != "user":
        release()
        raise HTTPException(status_code=409, detail="沒有需要重新產生的回答")
    try:
        knowledge = await _with_knowledge(conn, user, conv, history, settings, kb)
        llm = await run_in_threadpool(load_llm_settings, conn, settings)
    except BaseException:
        release()
        raise
    return await ndjson_reply(
        client,
        history,
        conversation_id,
        settings.db_path,
        release,
        knowledge.sources,
        system=_combine_system(llm.system_prompt, knowledge.prompt),
        temperature=llm.temperature,
        images=knowledge.images,
    )

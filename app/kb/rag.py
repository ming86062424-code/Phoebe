import io
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.kb.index import Source
from app.kb.parsers.base import IMAGE_NAME

logger = logging.getLogger(__name__)

NO_CONTEXT_NOTE = "（知識庫中沒有找到相關內容）"
# Longest edge of an image sent to the chat model; page images are already this size.
MAX_IMAGE_EDGE = 1536


@dataclass(frozen=True)
class SourceImage:
    source_number: int  # the [編號] of the source the image belongs to (1-based)
    data: bytes  # PNG


def _where(source: Source) -> str:
    return "全文" if source.page == 0 else f"第 {source.page} 頁"


def _image_lines(images: list[SourceImage]) -> str:
    mapping = "\n".join(
        f"附圖 {i} → [{image.source_number}]" for i, image in enumerate(images, start=1)
    )
    return f"\n\n附圖依序對應參考資料 [編號]：\n{mapping}"


def build_system_prompt(
    kb_name: str, sources: list[Source], images: list[SourceImage] | None = None
) -> str:
    if sources:
        blocks = "\n\n".join(
            f"[{i}] {s.filename} {_where(s)}\n{s.text}" for i, s in enumerate(sources, start=1)
        )
    else:
        blocks = NO_CONTEXT_NOTE
    return (
        f"你是助理，正在根據使用者的知識庫「{kb_name}」回答問題。\n"
        "請優先依據下列參考資料回答，並在用到的句子後以 [編號] 標註來源。"
        "若參考資料不足以回答，請明確說明知識庫中沒有相關資訊，再視情況補充一般知識並註明。\n\n"
        f"參考資料：\n{blocks}" + (_image_lines(images) if images else "")
    )


def _png_for_model(path: Path) -> bytes:
    """The file's bytes, or a PNG downscaled to MAX_IMAGE_EDGE when it is larger."""
    from PIL import Image

    from app.kb.parsers.images import check_pixels, fit, to_rgb

    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        check_pixels(image.width, image.height)
        if max(image.size) <= MAX_IMAGE_EDGE:
            return data
        image.load()
        smaller = fit(to_rgb(image), MAX_IMAGE_EDGE)
    out = io.BytesIO()
    smaller.save(out, format="PNG")
    return out.getvalue()


def load_source_images(
    sources: list[Source], doc_dir: Callable[[int], Path], limit: int
) -> list[SourceImage]:
    """Images for the chat model, in source (hybrid score) order: each source's page image, then
    its cropped images, until `limit` are loaded. Missing or unreadable files are skipped,
    and an image already taken for an earlier source is not repeated (it keeps that
    earlier source's number).

    `doc_dir(document_id)` is the document's image folder.
    """
    found: list[SourceImage] = []
    if limit <= 0:
        return found
    seen: set[tuple[int, str]] = set()  # an image cited by several sources is sent once
    for number, source in enumerate(sources, start=1):
        refs = ([source.page_image] if source.page_image else []) + list(source.images)
        for ref in refs:
            if len(found) >= limit:
                return found
            name = ref.get("name") if isinstance(ref, dict) else None
            if not isinstance(name, str) or not IMAGE_NAME.fullmatch(name):
                logger.warning("略過名稱不合法的參考圖片：%r", name)
                continue
            if (source.document_id, name) in seen:
                continue
            seen.add((source.document_id, name))
            path = doc_dir(source.document_id) / name
            try:
                data = _png_for_model(path)
            except FileNotFoundError:
                logger.warning("參考圖片不存在，略過：文件 %s 的 %s", source.document_id, name)
                continue
            except Exception as e:  # unreadable image: answer without it
                logger.warning(
                    "參考圖片無法讀取，略過：文件 %s 的 %s（%s）", source.document_id, name, e
                )
                continue
            found.append(SourceImage(number, data))
    return found

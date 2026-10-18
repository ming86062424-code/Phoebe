from app.kb.index import Source

NO_CONTEXT_NOTE = "（知識庫中沒有找到相關內容）"


def build_system_prompt(kb_name: str, sources: list[Source]) -> str:
    if sources:
        blocks = "\n\n".join(
            f"[{i}] {s.filename} 第 {s.page} 頁\n{s.content}" for i, s in enumerate(sources, start=1)
        )
    else:
        blocks = NO_CONTEXT_NOTE
    return (
        f"你是助理，正在根據使用者的知識庫「{kb_name}」回答問題。\n"
        "請優先依據下列參考資料回答，並在用到的句子後以 [編號] 標註來源。"
        "若參考資料不足以回答，請明確說明知識庫中沒有相關資訊，再視情況補充一般知識並註明。\n\n"
        f"參考資料：\n{blocks}"
    )

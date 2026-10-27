"""The running Phoebe system as the app under test: one knowledge base of the 25 md files in
test_data/rag, asked through the real chat API (the same retrieval and prompt a user gets)."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from scripts.benchmark import Bench

ROOT = Path(__file__).resolve().parent.parent
RAG_DIR = ROOT / "test_data" / "rag"


@dataclass
class Reply:
    answer: str = ""
    sources: list[dict] = field(default_factory=list)
    error: str | None = None


class LiveRag(Bench):
    def setup(self) -> dict:
        """Create the knowledge base and upload every md file; returns {file: status}."""
        files = sorted(RAG_DIR.glob("*/*.md"))
        if not files:
            raise FileNotFoundError(f"{RAG_DIR} 沒有 md 檔")
        # the parser is for PDFs and images; md files always use the text parser
        self.kb_id = self.create_kb(f"RAG 評測 {time.strftime('%m%d-%H%M')}", "liteparse")
        return {f.relative_to(RAG_DIR).as_posix(): self.upload(self.kb_id, f)["status"] for f in files}

    def ask(self, question: str) -> Reply:
        """One new conversation per question: retrieval never sees an earlier turn."""
        cid = self.c.post("/api/conversations").json()["id"]
        self.conversations.append(cid)
        self.c.patch(f"/api/conversations/{cid}", json={"kb_id": self.kb_id}).raise_for_status()
        reply, parts = Reply(), []
        with self.c.stream("POST", f"/api/conversations/{cid}/messages", json={"content": question}) as r:
            if r.status_code != 200:
                # e.g. 503 「Gemini 配額已用完或請求太頻繁」: the detail tells a retry apart
                r.read()
                raise RuntimeError(f"對話失敗 {r.status_code}：{r.text}")
            for line in r.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event["type"] == "sources":
                    reply.sources = event["sources"]
                elif event["type"] == "content":
                    parts.append(event["text"])
                elif event["type"] == "error":
                    reply.error = event.get("detail") or "對話失敗"
        reply.answer = "".join(parts)
        return reply

    def environment(self) -> dict:
        llm = self.get("/api/settings")
        search = self.get("/api/search-settings")
        provider = llm["llm_provider"]
        temperature = llm.get("temperature")
        return {
            "聊天模型": f"{provider} / {llm[f'{provider}_model']}，temperature {'模型預設' if temperature is None else temperature}",
            "檢索": f"前 {search['top_k']} 段，向量權重 {search['vector_weight']}",
        }

# 知識庫（RAG）— 設計規格

- 日期：2026-10-16
- 狀態：已確認（使用者授權直接開發）
- 前一版規格：`docs/superpowers/specs/2026-10-16-app-shell-design.md`

> **之後的變更**（2026-10-27 補註）：以下內容是當時的設計，保留作為紀錄。後來改變的部分：
>
> - 不再使用 LlamaIndex 與 Chroma：向量庫改為 Milvus 混合查詢（`2026-10-19-milvus-hybrid-design.md`），embedding 改由獨立的 embedder 服務產生（`2026-10-19-multimodal-kb-design.md`）。
> - 不再用 `SentenceSplitter` 切塊：PDF 每頁一個 chunk，txt／md 依段落或標題切塊（`docs/parsers.md`）。
> - 解析器改成**知識庫層級的設定**，不在上傳時選；支援的檔案類型擴充為 PDF、txt、md、圖檔。
> - `app/kb/worker.py` 改名為 `app/kb/ingest.py`（`2026-10-22-ingest-concurrency-design.md`）。

## 1. 目的與範圍

讓使用者上傳 PDF 建立個人知識庫，並在 AI 對話中選用知識庫，回答附上來源（檔名＋頁碼）。

**需求（使用者確認）**
1. 以 LlamaIndex 作為知識庫框架（文件、切塊、Embedding、向量庫、檢索）
2. PDF 解析器在上傳時選擇：`liteparse`（預設，快速、內建 Tesseract OCR）或 `LightOnOCR-3-0.8B`（transformers 內嵌，GPU）
3. 新功能頁「知識庫」：建立／改名／刪除知識庫、上傳與刪除文件、看處理進度、檢索測試
4. AI 對話可為每個對話選擇一個知識庫；回答下方顯示來源
5. 知識庫屬於個別使用者，彼此不可見
6. Embedding 使用 Ollama `embeddinggemma-2`

**明確不做（YAGNI）**：非 PDF 格式、重排序／混合檢索、知識庫分享、更換 Embedding 模型後自動重建索引、OCR 版面框選（grounding）。

## 2. 技術選型

| 項目 | 選擇 |
|---|---|
| 框架 | `llama-index-core` |
| Embedding | `llama-index-embeddings-ollama`，模型 `embeddinggemma-2` |
| 向量庫 | Chroma（`chromadb.PersistentClient`，`data/chroma`），每個知識庫一個 collection `kb_<id>` |
| 切塊 | `SentenceSplitter(chunk_size=512, chunk_overlap=64)`，以頁為單位切，保留頁碼 |
| 解析 | `liteparse`（Markdown 輸出）；LightOnOCR 用 `transformers` + `pypdfium2` |
| OCR 相依 | 可選的 uv 群組 `ocr`（torch CUDA 12.6、transformers），未安裝時介面顯示不可用 |

**回答生成**沿用既有的 `OllamaClient` 串流（保留思考過程即時顯示）；LlamaIndex 負責檢索，檢索結果組成 system prompt。

## 3. 後端架構

```
app/kb/
├── store.py          # SQLite：knowledge_bases、kb_documents CRUD
├── index.py          # KnowledgeIndex：Chroma + LlamaIndex 寫入／刪除／檢索
├── worker.py         # IngestWorker：背景執行緒依序處理文件
├── services.py       # KbServices：組合 index、parsers、worker，掛在 app.state.kb
├── rag.py            # 檢索結果 → system prompt 與 sources
└── parsers/
    ├── base.py       # PageText、Parser 介面
    ├── liteparse_parser.py
    └── lightonocr.py
app/routes/kb.py      # /api/kbs、/api/parsers
```

**資料表**
- `knowledge_bases(id, user_id, name, description, created_at, updated_at)`
- `kb_documents(id, kb_id, filename, parser, status, error, progress_done, progress_total, chunk_count, size_bytes, created_at, updated_at)`；status ∈ `queued | parsing | indexing | ready | failed`
- `conversations.kb_id`（可為 NULL，刪除知識庫時設為 NULL）、`messages.sources`（JSON 文字，可為 NULL）；舊資料庫以 `ALTER TABLE ADD COLUMN` 遷移
- 原始檔：`data/kb/<kb_id>/<doc_id>.pdf`

**背景處理**：單一執行緒佇列。`queued → parsing（逐頁進度）→ indexing → ready`；例外則 `failed` 並記錄訊息，可重試。啟動時把未完成的文件重新排入。佇列清空後呼叫解析器的 `release()`（OCR 釋放 VRAM）。OCR 開始前請 Ollama 以 `keep_alive=0` 卸載常駐模型。文件在處理中被刪除時，處理完成後清掉向量。

**API**
| 方法 | 路徑 | 說明 |
|---|---|---|
| GET | `/api/parsers` | `[{id, label, available, reason}]` |
| GET/POST | `/api/kbs` | 列表（含文件數）／建立 `{name, description}` |
| PATCH/DELETE | `/api/kbs/{id}` | 改名／刪除（含檔案與向量） |
| GET | `/api/kbs/{id}/documents` | 文件列表 |
| POST | `/api/kbs/{id}/documents` | multipart `file` + `parser`；僅 PDF、≤ 50MB |
| DELETE | `/api/kbs/{id}/documents/{doc_id}` | 刪除 |
| POST | `/api/kbs/{id}/documents/{doc_id}/retry` | 失敗後重試 |
| GET | `/api/kbs/{id}/search?q=` | 檢索測試，回傳 sources |
| PATCH | `/api/conversations/{id}` | `{kb_id: int \| null}` |

**聊天整合**：對話有 `kb_id` 時，以最後一則使用者訊息檢索 top-5，於歷史最前面加入 system 訊息；串流第一個事件為 `{"type":"sources","sources":[...]}`；sources 隨助理訊息存檔。source：`{document_id, filename, page, score, text}`（text 截 400 字）。檢索失敗（Ollama 無法連線）回 503。

## 4. 前端

- 功能註冊表新增 `{ id: "kb", label: "知識庫", path: "/kb" }`；路由 `#/kb`、`#/kb/:id`
- 知識庫頁：左欄知識庫列表（新增、刪除），右側為名稱（可改名）、上傳區（檔案＋解析器下拉）、文件表格（狀態、進度、區塊數、重試、刪除；有未完成文件時每 2 秒輪詢）、檢索測試框
- 聊天頁：輸入框上方下拉選擇知識庫（「不使用知識庫」為預設）；助理訊息下方可展開「來源」

## 5. 測試

- 後端：MockEmbedding + 假解析器 + 暫存 Chroma；worker 以同步模式測試；liteparse 用 `tests/pdf_factory.py` 產生的 PDF 實測
- LightOnOCR：不進單元測試；`scripts/ocr_smoke.py` 手動驗證
- 前端：Vitest（知識庫頁、來源顯示、知識庫選擇）

# 多模態知識庫與引用介面 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 知識庫支援 PDF／txt／md／圖檔，PDF 每頁一個 chunk；LiteParse 以整頁圖片產生向量、LightOnOCR 以官方 grounding 裁出圖片；embedding 統一由 sentence-transformers 服務產生；LLM 回答時看得到圖片；引用卡片可收合展開、顯示分數並排序。

**Architecture:** 新增 `embedder` 服務（FastAPI + sentence-transformers CPU float32）；`KnowledgeIndex` 改用 `EmbedderClient`；解析器輸出 `ParsedChunk`（文字、頁面圖、裁切圖）；Milvus 新增 `kind`／`images`／`page_image` 欄位；圖片以受保護的 API 提供；聊天把前幾個 source 的圖片附給 LLM。

**Tech Stack:** FastAPI、sentence-transformers、torch（CPU）、pypdfium2、Pillow、pymilvus、React + TypeScript

**Spec:** `docs/superpowers/specs/2026-10-19-multimodal-kb-design.md`

## Global Constraints

- embedding 模型 `google/embeddinggemma-2`，768 維、L2 normalize、CPU float32（**不可 float16**）；前綴：query `task: search result | query: {q}`、文件 `title: {title} | text: {content}`、圖片無前綴、圖文混合 `<|image|> {text}`
- 設定：`EMBEDDER_URL`（`http://localhost:8002`；Docker 內固定 `http://embedder:8000`）、`KB_MAX_IMAGES=4`；embedder 服務端：`EMBED_MODEL_ID=google/embeddinggemma-2`、`EMBED_IMAGE_TOKENS=280`；移除 `EMBED_MODEL`
- PDF 每頁一個 chunk；txt／md 段落合併至 ≤ 1500 字；圖檔一張一個 chunk
- LiteParse 頁面圖：400 DPI 渲染後縮到最長邊 1536 px，`page_{n}.png`；LightOnOCR 裁切圖 `img_{page}_{i}.png`；圖片名稱白名單 `^(page|img)_[0-9]+(_[0-9]+)?\.png$`
- 上傳副檔名：`.pdf .txt .md .png .jpg .jpeg .webp`；錯誤訊息（逐字）：「不支援的檔案類型（支援 PDF、txt、md、png、jpg、jpeg、webp）」、「無法讀取檔案內容」、「無法連線到 embedding 服務」、「知識庫結構已更新，請重建知識庫（刪除 collection 後重新處理文件）」
- 依賴與映像版本固定且發布超過兩週；新增依賴以 `uv add`
- 單元測試不下載模型、不需 Milvus／embedder；整合測試分別標 `milvus`、`embedder`
- Windows Git Bash；uv `/c/Users/anny3/AppData/Roaming/Python/Python312/Scripts`；Node `/c/Program Files/nodejs`；`PYTHONUTF8=1`；LF 換行；繁體中文
- 每個 commit 結尾：空一行後 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

1. **掃描 PDF 頁數很多（例如 200 頁）**：期望每頁圖片 embedding 分批送 embedder，不一次塞進記憶體或單一請求。→ Task 4 `test_liteparse_embeds_pages_in_batches`
2. **惡意圖片名稱（`../app.db`、`%2e%2e`）**：期望 404，不能讀到 kb 資料夾外的檔案。→ Task 5 `test_image_api_rejects_traversal`
3. **txt 檔是 Big5／含 BOM**：期望 UTF-8-SIG 可讀；非 UTF-8 給「無法讀取檔案內容」而非 500。→ Task 3 `test_text_upload_encodings`
4. **聊天模型不支援圖片**：期望不帶圖重試成功，不讓使用者看到錯誤。→ Task 5 `test_retries_without_images_when_model_rejects_them`
5. **引用很多且有長 chunk**：期望預設收合、只顯示 120 字，不會撐爆訊息區。→ Task 6 `cards are collapsed by default and show a 120-char preview`

---

### Task 1: Embedding 服務

**Files:**
- Create: `embedder/pyproject.toml`、`embedder/uv.lock`（獨立 uv 專案：fastapi、uvicorn、sentence-transformers、torch CPU、pillow；torch 走 CPU index）、`embedder/app.py`、`embedder/tests/test_app.py`、`docker/embedder.Dockerfile`
- Modify: `docker/docker-compose.yml`（`embedder` 服務：127.0.0.1:8002:8000、`hf-cache` volume、healthcheck `/health`、start_period 夠長）、`docker/README.md`

**Interfaces:**
- Produces: `POST /embed`（規格 §2.1 的 items 格式）→ `{"embeddings": [[float]*768]}`；`GET /health`；`create_app(model=None)` 可注入假模型

- [ ] **Step 1: tests**（假模型記錄收到的 encode 參數）：前綴正確、圖片不加前綴、image_text 使用 `<|image|>`、`max_soft_tokens`、輸出 L2 normalized、批次、未載入時 `/health` 503、錯誤輸入 422
- [ ] **Step 2: Run** `cd embedder && uv run pytest -q` → FAIL
- [ ] **Step 3: Implement**（以已安裝 sentence-transformers 原始碼確認多模態 encode 的輸入格式與 `processing_kwargs`；dtype float32、device cpu）
- [ ] **Step 4: Run** → PASS；`cd docker && docker compose config -q`
- [ ] **Step 5: Commit** `feat: sentence-transformers embedding service for text and images`

---

### Task 2: App 改用 embedder

**Files:**
- Create: `app/kb/embedder_client.py`
- Modify: `app/kb/index.py`、`app/kb/services.py`、`app/kb/migrate_chroma.py`、`app/config.py`、兩份 `.env.example`、`docker/docker-compose.yml`（web/worker `EMBEDDER_URL`、依賴 embedder healthy）、`pyproject.toml`（移除 `llama-index-embeddings-ollama`）、測試

**Interfaces:**
- Produces: `EmbedderClient(url, transport=None)`：`embed_queries(list[str])`、`embed_documents(title: str, texts: list[str])`、`embed_images(list[bytes])`、`embed_image_texts(list[tuple[bytes, str]])` → `list[list[float]]`；批次 ≤ 16；逾時 120 s；連線錯誤 → `KnowledgeStoreError("無法連線到 embedding 服務")`
- `KnowledgeIndex(store, embedder)`：行為不變（目前文字流程），查詢用 `embed_queries`，文件用 `embed_documents(filename, texts)`
- 測試替身 `tests/fakes_embedder.FakeEmbedder`（確定性向量）

- [ ] Step 1 tests（client 請求格式／批次／錯誤；index 改用替身後既有測試照過）→ Step 2 FAIL → Step 3 實作 → Step 4 `uv run pytest -q` PASS（含 `--group migrate`）
- [ ] Step 5: Commit `feat: use the embedding service for queries and documents`

---

### Task 3: 檔案類型、ParsedChunk、Milvus 新欄位

**Files:**
- Modify: `app/kb/parsers/base.py`（`ImageRef`、`ParsedChunk`）、`app/kb/parsers/text_parser.py`（新：txt／md 切塊）、`app/routes/kb.py`（副檔名、內容檢查、文字檔自動用 text parser；`file_path` 改為保留副檔名）、`app/kb/services.py`、`app/kb/worker.py`、`app/kb/index.py`（`add_document(kb_id, doc_id, filename, chunks: list[ParsedChunk])`；依 kind 選 embedding）、`app/kb/chunk_store.py`、`app/kb/milvus_store.py`（`kind`／`images`／`page_image` 欄位、schema 檢查）、`app/db.py`（如需記錄檔案副檔名）、`app/kb/migrate_chroma.py`
- Test: 對應測試

**Interfaces:**
- `ImageRef(name: str, width: int, height: int)`；`ParsedChunk(page: int, text: str, images: list[ImageRef] = [], page_image: ImageRef | None = None, kind: Literal["text","page_image"] = "text")`
- Parser 協定：`parse(path: str, out_dir: Path, on_progress) -> list[ParsedChunk]`
- `Source` 新增 `images: list[dict]`、`page_image: dict | None`、`kind: str`；`text` = 完整 chunk 文字

- [ ] Step 1 tests：`test_text_upload_encodings`（UTF-8、UTF-8-SIG 可；Big5 → 422「無法讀取檔案內容」）、txt／md 切塊（標題分段、≤1500）、不支援副檔名 422、圖片內容檢查、schema 缺欄位錯誤、Source 新欄位與完整文字
- [ ] Step 2 FAIL → Step 3 實作 → Step 4 `uv run pytest -q` 與 `-m milvus` PASS
- [ ] Step 5: Commit `feat: text and image uploads with page-level chunks and image fields`

---

### Task 4: LiteParse 頁面圖片向量、LightOnOCR grounding

**Files:**
- Modify: `app/kb/parsers/liteparse_parser.py`、`app/kb/parsers/lightonocr.py`、`app/kb/parsers/ocr_engines.py`（grounding 提示）、`app/kb/index.py`（`page_image` chunk 用 `embed_image_texts`）
- Test: `tests/test_parsers.py`、`tests/test_kb_index.py`

**Interfaces:**
- LiteParse：每頁 → `ParsedChunk(kind="page_image", page_image=ImageRef(page_{n}.png), text=該頁文字)`；圖檔視為單頁（圖檔本身另存為 `page_1.png`）
- LightOnOCR：請求 grounding；`parse_grounding(text) -> list[Block(label, bbox, text)]`；裁切 `image` 區塊；丟棄 header／footer／page_number；chunk 文字為 markdown，圖片以 `![圖片 i](img:{name}) 描述` 表示
- embedding 分批（每批 ≤ 8 頁）

- [ ] Step 1 tests：`test_liteparse_embeds_pages_in_batches`、頁面圖尺寸（最長邊 1536）、grounding 解析與座標換算（0–1000 → 像素）、裁切檔案存在、頁首頁尾頁碼被去掉、圖檔單頁
- [ ] Step 2 FAIL → Step 3 實作（LiteParse 截圖／圖片 API 以已安裝原始碼為準）→ Step 4 PASS
- [ ] Step 5: Commit `feat: page-image vectors for LiteParse and grounding crops for LightOnOCR`

---

### Task 5: 圖片 API 與 LLM 看圖

**Files:**
- Modify: `app/routes/kb.py`（圖片端點）、`app/llm/base.py`、`app/llm/ollama.py`、`app/llm/gemini.py`、`app/routes/chat.py`、`app/kb/rag.py`、`app/config.py`
- Test: `tests/test_kb_api.py`、`tests/test_ollama_client.py`、`tests/test_gemini_client.py`、`tests/test_chat.py`

**Interfaces:**
- messages 中 user 訊息可帶 `images: list[bytes]`；Ollama → base64 `images`；Gemini → `Part.from_bytes(..., mime_type="image/png")`
- 聊天：依 source 排序取圖（page_image 優先，再 images），最多 `KB_MAX_IMAGES`；system 加「附圖依序對應參考資料 [編號]」
- Ollama 回應錯誤訊息含 image／vision 不支援 → 不帶圖重試一次

- [ ] Step 1 tests：`test_image_api_rejects_traversal`、權限（他人 KB 404）、名稱白名單、圖片傳遞格式（兩供應商）、最多 4 張、`test_retries_without_images_when_model_rejects_them`
- [ ] Step 2 FAIL → Step 3 → Step 4 PASS
- [ ] Step 5: Commit `feat: serve knowledge images and send them to the chat model`

---

### Task 6: 前端引用卡片與上傳

**Files:**
- Create: `frontend/src/features/kb/SourceCard.tsx`（含 lightbox）與測試
- Modify: `frontend/src/api/types.ts`（Source 新欄位）、聊天訊息的參考資料區、`KbDetail.tsx`（檢索測試沿用卡片；上傳 accept 與文字檔隱藏解析器）

- [ ] Step 1 tests：依 score 降冪排序、`cards are collapsed by default and show a 120-char preview`、展開顯示全文與縮圖、`img:` 標記渲染為圖片、lightbox、分數格式（相關度 4 位、向量 3 位、關鍵字 2 位、缺值「—」）、上傳 accept
- [ ] Step 2 FAIL → Step 3 → Step 4 `npm test && npm run lint && npm run typecheck && npm run build` PASS
- [ ] Step 5: Commit `feat: collapsible scored citation cards with images`

---

### Task 7: 重新處理指令、文件與實測

**Files:**
- Create: `app/kb/reprocess_all.py`（所有未刪除文件設為 `queued`；`--dry-run`）
- Modify: README、docker/README、spec（部署 spec 的 embedding 說明）
- Test: `tests/test_reprocess_all.py`

- [ ] Step 1–4：TDD reprocess；文件（embedder、重建步驟）
- [ ] Step 5: Commit `feat: reprocess-all command and docs for the multimodal knowledge base`
- [ ] Step 6（controller）：Docker 重建、刪 collection、reprocess；上傳含圖 PDF（LiteParse 與 LightOnOCR）、txt、md、png；以圖片內容提問，確認引用與圖片顯示；`-m milvus`、`-m embedder` 整合測試

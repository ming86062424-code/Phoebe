# 多模態知識庫與引用介面 — 設計規格

- 日期：2026-10-19
- 子專案：4 / 4（部署 → LLM 供應商 → Milvus 混合查詢 → **多模態知識庫**）
- 建立在子專案 3（`feature/milvus`，PR #8）之上
- 狀態：使用者授權直接開發

> **之後的變更**（2026-10-27 補註）：以下內容是當時的設計，保留作為紀錄。後來改變的部分：
>
> - 引用卡片的分數改為加權相加的分數，不再是 RRF（見 `2026-10-19-milvus-hybrid-design.md` 的後續變更）。
> - 「embedding 服務的 GPU 版」後來做了（`2026-10-24-embedder-gpu-mode-design.md`）。
> - 解析器改成知識庫層級的設定，不在上傳時選。
> - Chroma 遷移指令已移除；`worker` 已改名為 `ingest`。

## 1. 目的與範圍

**需求（使用者確認）**
1. 知識庫能處理文件中的圖片；LLM 回答時參考得到圖片；介面顯示圖片本身（不是佔位符）
2. 除了 PDF，也能上傳 txt、md、圖檔
3. 聊天的引用 chunk 預設收合，按鈕展開看完整內容
4. 引用 chunk 顯示分數並依分數排序
5. **PDF 每頁一個 chunk**（預設）
6. 兩種解析器各自的做法：
   - **LiteParse**：每頁 PDF 渲染成圖片，**以整頁圖片產生向量**；BM25 用 LiteParse 抽出的該頁文字
   - **LightOnOCR**：照官方用法，**grounding 模式**；依區塊座標裁出圖片；chunk 為該頁 markdown（去掉頁首、頁尾、頁碼）
7. embedding 改由**獨立的 embedding 服務**（sentence-transformers、CPU、float32）產生；**查詢、文字、圖片全部用同一個服務**，不再用 Ollama 做 embedding

**事實（已驗證）**
- `embeddinggemma-2` 支援圖片，但 Ollama 0.40.6 `/api/embed` 會忽略圖片（實測：帶圖與不帶圖向量相同）→ 必須用 sentence-transformers
- HF `google/embeddinggemma-2` 未 gated、Apache 2.0；共用 8192 token context；圖片預設 280 soft tokens；輸出 768 維；模型卡明言**不可用 float16**（RTX 2060 不支援 bf16，所以用 CPU float32）
- 查詢前綴 `task: search result | query: {q}`；文件 `title: {title} | text: {content}`；圖片不加前綴

**明確不做**：音訊／影片、圖片以外的多模態檢索 UI（以圖搜圖）、rerank、embedding 服務的 GPU 版、Ollama embedding 相容模式。

## 2. 架構

### 2.1 Embedding 服務 `embedder/`
- 獨立的 FastAPI 小服務（`embedder/app.py`，依賴 `sentence-transformers`、`torch`（CPU wheel）、`pillow`），Docker 服務 `embedder`（映像檔獨立 build；模型快取 volume `hf-cache`）
- `POST /embed`：`{"items": [{"kind": "query", "text": "..."} | {"kind": "document", "title": "...", "text": "..."} | {"kind": "image", "image_b64": "<png>"} | {"kind": "image_text", "image_b64": "...", "text": "..."}]}` → `{"embeddings": [[...768], ...]}`（L2 normalized）
  - 前綴依官方：query 與 document 套用上述格式；image 不加前綴；image_text 為 `{"text": "<|image|> " + text, "image": img}`
  - 批次處理；圖片 `max_soft_tokens` 預設 280（設定 `EMBED_IMAGE_TOKENS`）
- `GET /health`：模型載入完成才回 200
- 本機快速測試：`cd docker && docker compose up -d embedder`（與 milvus 一樣只綁 127.0.0.1，埠 8002）
- App 端 `app/kb/embedder_client.py`：`EmbedderClient(url)` 提供 `embed_queries`、`embed_documents(title, texts)`、`embed_images(images)`、`embed_image_texts`；連不上 → `KnowledgeStoreError("無法連線到 embedding 服務")`
- 設定：`EMBEDDER_URL`（預設 `http://localhost:8002`，Docker 內 `http://embedder:8000`）；移除 `EMBED_MODEL`（模型名改在 embedder 服務設定 `EMBED_MODEL_ID=google/embeddinggemma-2`）

### 2.2 解析器輸出
- `PageText` 擴充為 `ParsedChunk(page: int | None, text: str, images: list[ImageRef], page_image: ImageRef | None)`；`ImageRef(name: str, width, height)`（檔案存在 `kb_dir/<kb>/<doc>/` 下）
- 檔案類型與解析器：
  | 類型 | 可選解析器 | chunk |
  |---|---|---|
  | PDF | LiteParse、LightOnOCR | 每頁一個 |
  | 圖檔（png、jpg、jpeg、webp） | LiteParse、LightOnOCR | 一張一個（視為單頁） |
  | txt | 文字（自動） | 依段落合併至 ≤ 1500 字 |
  | md | 文字（自動） | 依標題分段，超過 1500 字再依段落切 |
- **LiteParse（PDF／圖檔）**：每頁渲染成圖（400 DPI 再縮到最長邊 1536 px，存成 `page_{n}.png`）；文字用 LiteParse 抽取；向量 = `image_text`（整頁圖＋該頁文字，讓圖文一起進向量）；`page_image` 指向整頁圖
- **LightOnOCR（PDF／圖檔）**：官方 grounding（請求帶 `{"type":"text","text":"grounding"}`、temperature 0.1）；解析 `![label](x1,y1,x2,y2) text` 區塊（座標 0–1000）；`image` 區塊依座標從送進模型的頁面圖裁切存成 `img_{page}_{i}.png`，chunk 文字中以 `![圖片 i](img:{name}) 描述` 表示；捨棄 `header`、`footer`、`page_number`；向量 = `document`（頁面 markdown 文字）；`images` 為裁出的圖
- 文字檔：向量 = `document`；無圖片
- 上傳限制：副檔名白名單、內容檢查（PDF magic、圖片可被 PIL 開啟、文字可用 UTF-8／UTF-8-SIG 解碼，否則 422「無法讀取檔案內容」）

### 2.3 Milvus
- collection 新增欄位：`images` VARCHAR（JSON 陣列 `[{"name","w","h"}]`）、`page_image` VARCHAR（可空字串）、`kind` VARCHAR（`page_image`、`text`）；`page` 對文字檔為 0
- `ensure_ready` 增加 schema 欄位檢查：缺欄位 → `KnowledgeStoreError("知識庫結構已更新，請重建知識庫（刪除 collection 後重新處理文件）")`
- `Source` / `to_dict()` 新增 `images`、`page_image`、`kind`；引用給前端的 `text` 改為**完整 chunk 文字**（收合／展開由前端處理），不再截 400 字

### 2.4 圖片 API
- `GET /api/kbs/{kb_id}/documents/{doc_id}/images/{name}`：需登入、需擁有該 KB；`name` 只允許 `^(page|img)_[0-9]+(_[0-9]+)?\.png$`；回 `image/png`；找不到 404

### 2.5 LLM 參考圖片
- `ChatClient.stream_chat` 的 messages 允許 `images: list[bytes]`（只用在最後一則 user 訊息）
  - Ollama：message 的 `images`（base64）
  - Gemini：`Part.from_bytes(data, mime_type="image/png")`
- 知識庫回答時：依排序取前面 source 的圖片（`page_image` 優先，接著 `images`），最多 `KB_MAX_IMAGES`（預設 4）張，附在最後一則 user 訊息；系統提示說明「附圖依序對應參考資料 [編號]」
- 模型不支援圖片時（例如換成純文字模型）：Ollama 會回錯誤 → 若錯誤訊息顯示不支援圖片，改為不帶圖重試一次，並在 system 註明「（圖片未提供給模型）」

### 2.6 引用介面
- 聊天訊息下方「參考資料」區：依 `score`（RRF）由高到低排序
- 每張卡片預設收合：`[編號] 檔名 · 第 N 頁`（文字檔顯示「全文」）＋分數列「相關度 0.0328　向量 0.812　關鍵字 2.55」（缺值顯示「—」）＋文字前 120 字
- 「展開」按鈕：顯示完整 chunk 文字（`pre-wrap`；`![圖片 i](img:name) …` 標記以實際圖片縮圖呈現）、`page_image` 縮圖、所有 `images` 縮圖；點縮圖以 lightbox 放大；「收合」按鈕
- 知識庫頁的檢索測試沿用同一個卡片元件
- 上傳介面：接受 `.pdf,.txt,.md,.png,.jpg,.jpeg,.webp`；文字檔隱藏解析器選擇

### 2.7 重建
- 因 embedding 模型 runtime 與 schema 都改變，既有 collection 需重建：
  - 刪除 collection → 重啟 web/worker →「全部重新處理」：`python -m app.kb.reprocess_all`（把所有未刪除的文件設為 `queued`，worker 依原解析器重新解析）
  - Chroma 遷移指令保留，但改用新的 embedder；遷移來的舊 chunk 只有文字（`kind=text`）

## 3. 錯誤處理
| 情境 | 行為 |
|---|---|
| embedder 連不上 | 索引：文件 failed「無法連線到 embedding 服務」；檢索：503「知識庫檢索失敗：無法連線到 embedding 服務」 |
| 不支援的檔案類型 | 422「不支援的檔案類型（支援 PDF、txt、md、png、jpg、jpeg、webp）」 |
| 圖片 API 路徑不合法 | 404 |
| 送圖給不支援圖片的模型 | 不帶圖重試一次 |

## 4. 測試
- embedder 服務：以假的 SentenceTransformer 測請求格式、前綴、正規化、批次、health；（不在單元測試下載模型）
- 整合（`@pytest.mark.embedder`，`EMBEDDER_URL` 可連時才跑）：文字 query 與相關文件 cosine 高於無關文件；頁面圖片（含「柴犬」字樣）與 query「日本的狗」相似度高於另一張圖
- 解析器：LiteParse 頁面渲染與 `page_image`；LightOnOCR grounding 解析（區塊、座標換算、裁切、捨棄頁首頁尾頁碼）；txt/md 切塊；圖檔當單頁；內容檢查
- Milvus：新欄位、schema 檢查
- 圖片 API：權限、名稱白名單、404
- LLM：Ollama／Gemini 圖片傳遞格式；最多 4 張；不支援圖片時重試
- 前端：卡片排序、收合／展開、分數顯示、圖片縮圖與放大、上傳接受新類型
- 實測（Docker）：上傳含圖片的 PDF（LiteParse 與 LightOnOCR 各一）、txt、md、png；以圖片內容的問題問答，回答引用正確且介面顯示圖片

## 5. 開發流程
- 分支 `feature/multimodal-kb`（從 `feature/milvus`），PR 目標 `feature/milvus`

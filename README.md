# Phoebe

在本機 Ollama 上運作的聊天應用（FastAPI + SQLite + React），附多模態知識庫（PDF、txt、md、圖檔；Milvus，BM25 + 向量混合檢索，文字與圖片共用同一個向量空間）。

## 事前準備

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Node 22 以上（前端建置與開發用）
- Ollama 已啟動，位於 http://localhost:11434
- 下載模型：`ollama pull gemma4:e4b`（評測另需 `ollama pull qwen3.5:4b`）
- 知識庫 Embedding 服務（`embedder`，以 Docker 執行）：`cd docker && docker compose up -d embedder`，位於 http://localhost:8002（見下方「Embedding 服務」）

## 安裝與執行

```
cp .env.example .env   # 依需要修改；環境變數優先於 .env
uv sync
uv run uvicorn app.main:create_app --factory --reload
```

前端需先建置才能由後端提供（見下方「前端」一節），再開啟 http://localhost:8000。

## 前端

前端位於 `frontend/`（React + TypeScript + Vite）。

安裝套件：

```
cd frontend && npm install
```

開發模式（兩個終端機各跑一個指令，開啟 http://localhost:5173）：

```
uv run uvicorn app.main:create_app --factory --reload
cd frontend && npm run dev
```

正式模式（建置後只需開 uvicorn，開啟 http://localhost:8000）：

```
cd frontend && npm run build
uv run uvicorn app.main:create_app --factory
```

注意：後端在啟動時才決定要不要提供 `frontend/dist`，所以第一次執行 `npm run build` 之後，必須重新啟動 uvicorn。

前端測試與檢查：

```
cd frontend && npm test
cd frontend && npm run lint
cd frontend && npm run typecheck
```

## 知識庫

側邊欄「知識庫」可建立知識庫、上傳文件；AI 對話上方的下拉選單可為每個對話選用一個知識庫。回答下方會列出引用卡片：每張卡片顯示檔名、頁碼、相關度分數與內容預覽（預設收合、只顯示約 120 字，可展開），有圖片的來源會附縮圖，點擊可放大。

支援的檔案類型：`.pdf`、`.txt`、`.md`、`.png`、`.jpg`、`.jpeg`、`.webp`。

- **PDF**：每頁一個 chunk（整頁文字，加上該頁的圖片資訊）。
- **txt／md**：txt 依段落合併、md 依標題分段，每個 chunk 至多 1500 字（需為 UTF-8，含 BOM 也可）；不需選解析器。
- **圖檔**：一張圖一個 chunk，直接以圖片向量檢索。

### Embedding 服務（embedder）

文字與圖片共用同一個 embedding 模型 `google/embeddinggemma-2`（768 維），由獨立的 `embedder` 服務提供（`embedder/`，sentence-transformers，CPU float32，不需 GPU）。應用程式以 `EMBEDDER_URL` 連線（本機預設 `http://localhost:8002`；Docker 內固定為 `http://embedder:8000`）。因此「用文字問、命中圖片內容」可以直接運作，不需要先把圖片轉成文字。

- 首次啟動會從 Hugging Face 下載模型（約 1–2 GB），第一次啟動會等很久；載入完成前 `GET /health` 回 503，檢索與解析會顯示「無法連線到 embedding 服務」。
- 本機啟動：`cd docker && docker compose up -d embedder`。細節與疑難排解見 [docker/README.md](docker/README.md)。

上傳 PDF 時可選擇解析器：

| 解析器 | 適合 | 說明 |
| --- | --- | --- |
| LiteParse | 文字型 PDF | 預設、速度快、不需 GPU；掃描頁會用內建 Tesseract（`chi_tra+eng`）辨識，但中文品質有限。每頁另存一張頁面圖（400 DPI 渲染後縮到最長邊 1536 px，`page_{n}.png`），向量由「整頁圖＋該頁文字」產生（**頁面圖向量**），圖表、照片與版面也能被檢索到 |
| LightOnOCR-3-0.8B | 掃描檔、表格、複雜版面 | 在本機 GPU 執行，輸出 Markdown 與各區塊座標（grounding）；RTX 2060 上每頁約 1–2 分鐘。依座標把頁面中的圖片**裁切**成獨立的圖（`img_{頁}_{序}.png`，捨棄頁首、頁尾與頁碼）；向量由頁面的 Markdown 文字產生，裁切圖附在該頁的引用卡片上，也會一併附給聊天模型 |

兩者差別：LiteParse 以「整頁圖」進入向量（圖片內容本身可被文字問題檢索到），LightOnOCR 以「文字」進入向量、另存「裁切出的單張圖」供引用與附圖（能精準顯示單張圖表）。圖檔（png、jpg、jpeg、webp）可選擇任一解析器，視為單頁。

LightOnOCR 需要額外的 `ocr` 套件群組（torch CUDA 12.6、transformers，約 3GB），並以該群組啟動伺服器：

```
uv sync --group ocr
uv run --group ocr uvicorn app.main:create_app --factory
```

未以 `--group ocr` 啟動時，介面會把 LightOnOCR 顯示為「不可用」。第一次使用會從 Hugging Face 下載模型（約 1.8GB）。OCR 開始前會請 Ollama 卸載常駐模型以騰出 VRAM，處理佇列清空後會釋放 OCR 模型。

手動驗證 OCR：

```
uv run --group ocr python -m scripts.ocr_smoke 路徑/檔案.pdf
```

文件在背景依序處理（排隊中 → 解析中 → 建立索引中 → 完成／失敗）；伺服器重啟後，未完成的文件會自動重新處理。

### 回答時附圖與 `KB_MAX_IMAGES`

回答時會把檢索到的參考圖片一併附給聊天模型（依參考資料順序，頁面圖優先，最多 `KB_MAX_IMAGES` 張，預設 4；最長邊超過 1536 px 會先縮小）。模型不支援圖片時，系統會自動不帶圖重試一次，使用者不會看到錯誤。

注意（Ollama 視覺模型）：4 張大圖可能超過預設的 context 長度（`num_ctx`）。若回答被截斷或失敗，請調低 `KB_MAX_IMAGES`（`0` 為不附圖），或調高 Ollama 的 `num_ctx`（例如在 Modelfile 設 `PARAMETER num_ctx 8192`，或設定環境變數 `OLLAMA_CONTEXT_LENGTH`）。

### 重新處理既有文件

升級自舊版（或更換 embedding 模型、解析設定）後，既有文件需要重新處理：先刪除 Milvus 的 collection，再把所有文件重新排入佇列。Docker 的完整步驟見 [docker/README.md](docker/README.md)；本機：

```
uv run python -m app.kb.reprocess_all --dry-run   # 只列出每個知識庫會排入幾份文件
uv run python -m app.kb.reprocess_all
```

指令會把所有「完成」與「失敗」的文件設為排隊中（清除錯誤與 chunk 數），由 worker 依原解析器重新處理；正在處理中的文件會略過，待其完成後再執行一次即可。

## 測試

單元測試（不需要 Milvus）：

```
uv run pytest
```

本機執行應用程式或整合測試需要 Milvus（以 Docker 執行）：`cd docker && docker compose up -d milvus`。整合測試標記為 `milvus`，連不上就自動略過：

```
MILVUS_URI=http://127.0.0.1:19530 uv run pytest -m milvus
```

評測（僅手動執行；需要真實的 Ollama，會跑數分鐘；分數僅供觀察，不是通過／失敗的門檻）：

```
uv run --group eval pytest evals -v
```

注意：單純執行 `uv sync` 會移除 eval、ocr 群組的套件；`uv run --group eval`／`uv run --group ocr` 會重新安裝。

## 設定（.env）

設定可放在專案根目錄的 `.env`（本機快速測試用，範本見 `.env.example`；Docker 部署改用 `docker/.env`，範本見 `docker/.env.example`）。本機可用 `ENV_FILE` 指定其他路徑。同名的環境變數優先於 `.env`。`.env` 已被 git 忽略；測試不會讀取它。

## 管理員與系統設定

- 管理員：在 `.env`（Docker 為 `docker/.env`）設定 `ADMIN_USERNAMES=alice,bob`，以逗號分隔，帳號大小寫需與登入帳號完全相符。修改 `.env` 後必須重新啟動服務才會生效。
- 只有管理員會在側邊欄看到「系統設定」頁，所有使用者的聊天都套用這份設定：
  - LLM 供應商：本機 Ollama 或 Google Gemini（未設定 `GEMINI_API_KEY` 時 Gemini 選項停用）
  - 模型：從目前提供者即時取得模型清單後選擇
  - 系統提示詞：最多 4000 字
  - temperature：可自訂，或使用模型預設值
- Embedding（知識庫）一律使用獨立的 `embedder` 服務（`EMBEDDER_URL`），不受 LLM 供應商影響。

### Gemini

1. 到 [Google AI Studio](https://aistudio.google.com/apikey) 以 Google 帳號登入並建立 API key。
2. 把金鑰填入 `.env` 的 `GEMINI_API_KEY=`（Docker 則填 `docker/.env`），可選填 `GEMINI_MODEL`（預設 `gemini-2.5-flash`），然後重新啟動。
3. 以管理員登入，到「系統設定」把LLM 供應商切換為 Gemini。

金鑰只存在環境設定中，不會出現在介面或資料庫。當 GPU 專用於 OCR（vLLM）時，建議聊天改用 Gemini（雲端），避免與 OCR 搶顯存。

## 環境變數

| 變數 | 預設值 | 說明 |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 位址 |
| `CHAT_MODEL` | `gemma4:e4b` | 聊天使用的模型（Ollama 預設值；可在「系統設定」頁覆寫） |
| `GEMINI_API_KEY` | 空 | Google Gemini API 金鑰；只能放在 `.env`／`docker/.env`，留空則「系統設定」中的 Gemini 選項停用 |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini 預設模型（可在「系統設定」頁覆寫） |
| `ADMIN_USERNAMES` | 空 | 管理員帳號，以逗號分隔（大小寫需完全相符） |
| `EVAL_JUDGE_MODEL` | `qwen3.5:4b` | 評測的裁判模型 |
| `DB_PATH` | `<專案根目錄>/data/app.db` | SQLite 資料庫路徑 |
| `SESSION_DAYS` | `7` | 登入工作階段有效天數 |
| `CHAT_THINK` | `true` | 是否讓模型輸出思考過程（`false`、`0`、`no`、`off` 為關閉，不分大小寫） |
| `EMBEDDER_URL` | `http://localhost:8002` | Embedding 服務（`embedder`）位址；Docker 內固定為 `http://embedder:8000`；更換模型後需重建知識庫 |
| `KB_DIR` | `<專案根目錄>/data/kb` | 上傳檔案與解析出的圖片存放位置 |
| `KB_TOP_K` | `5` | 每次回答檢索的段落數 |
| `KB_MAX_IMAGES` | `4` | 回答時附給模型的參考圖片上限（依參考資料順序，頁面圖優先；最長邊超過 1536 px 會先縮小）；`0` 不附圖。模型不支援圖片時會自動改為不帶圖重試一次 |
| `MAX_UPLOAD_MB` | `50` | 單一上傳檔案上限 |
| `OCR_MODEL` | `lightonai/LightOnOCR-3-0.8B` | LightOnOCR 模型（Hugging Face） |
| `LITEPARSE_OCR_LANGUAGE` | `chi_tra+eng` | LiteParse 內建 Tesseract 的語言 |
| `KB_WORKER_MODE` | `embedded` | 文件解析 worker：`embedded`（內嵌於 web）或 `external`（獨立 `python -m app.kb.worker_main`） |
| `KB_WORKER_POLL_SECONDS` | `2` | worker 輪詢間隔 |
| `KB_STALE_SECONDS` | `120` | 解析中逾時多久視為卡住並重新排入 |
| `KB_HEARTBEAT_SECONDS` | `30` | 處理中的文件刷新 `updated_at` 的間隔 |
| `REPLY_LOCK_TTL_SECONDS` | `600` | 對話回覆鎖逾時秒數 |
| `OCR_BACKEND` | `transformers` | `transformers` 或 `vllm` |
| `VLLM_BASE_URL` | `http://localhost:8001/v1` | vLLM 位址 |
| `VLLM_GPU_UTIL` | `0.35` | vLLM 顯存比例（僅 Docker） |
| `MILVUS_URI` | `http://localhost:19530` | Milvus 位址（快速測試：`cd docker && docker compose up -d milvus`；Docker 內固定為 `http://milvus:19530`） |
| `MILVUS_TOKEN` | 空 | Milvus 驗證 token（未啟用驗證則留空） |
| `MILVUS_COLLECTION` | `phoebe_chunks` | 存放知識庫段落的 collection 名稱 |
| `WEB_CONCURRENCY` | `2` | Gunicorn worker 數（僅 Docker，設在 `docker/.env`） |
| `PHOEBE_PORT` | `8000` | Docker 對外連接埠 |
| `FRONTEND_DIST` | `<專案根目錄>/frontend/dist` | 前端建置輸出資料夾，由後端提供；使用相對路徑時，會以啟動 uvicorn 的工作目錄為基準 |

## 部署

以 Docker Compose 部署（Gunicorn web、獨立解析 worker、Milvus，選用 vLLM），詳見 [docker/README.md](docker/README.md)：

```
cd docker
cp .env.example .env   # Docker 專用設定，與根目錄 .env 無關
docker compose up -d --build
```

## 資料位置

資料存放在 SQLite 檔案 `data/app.db`，知識庫檔案在 `data/kb/`（位於專案根目錄，與啟動目錄無關；可用 `DB_PATH` 覆寫）。`data/` 已被 git 忽略。向量與 BM25 索引存放在 Milvus（Docker 的 `milvus-data` volume）。舊版 Chroma 資料的遷移方式見 [docker/README.md](docker/README.md)。

## 專案結構

- `app/` — 後端（設定、資料庫、驗證、對話、Ollama 客戶端、路由）
- `app/kb/` — 知識庫（資料存取、解析器、索引、背景處理）
- `scripts/` — 手動驗證腳本
- `frontend/` — 前端（React + TypeScript + Vite）
- `tests/` — 單元測試
- `evals/` — 聊天品質評測（deepeval）

## 開發流程

所有變更都透過 feature 分支與 Pull Request 進行，由 repo 擁有者審查後合併，不直接推送到 `main`。

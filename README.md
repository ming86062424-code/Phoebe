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

## 本機開發模式（搭配 Docker 的模型服務）

改程式時不必每次重建 Docker 映像：web 與 ingest 直接在本機執行（改了程式會自動重新載入），Milvus、embedder、vLLM 這些重的服務繼續用 Docker 裡的。

**1. 啟動 Docker 裡的服務，並停掉 Docker 版的 web 與 ingest**（它們佔用 8000 埠，也會跟本機版搶工作）：

```
cd docker
docker compose up -d milvus embedder                # 必要：向量資料庫與 embedding
docker compose --profile lightonocr up -d vllm      # 選用：要測 LightOnOCR 時
docker compose stop web ingest
```

**2. 根目錄的 `.env`**（本機版讀這個檔，與 `docker/.env` 無關）至少設定：

```
MILVUS_URI=http://localhost:19530
EMBEDDER_URL=http://localhost:8002
# 與 Docker 版分開：本機版有自己的 SQLite，知識庫編號會和 Docker 版重複，
# 共用同一個 collection 會讓兩邊的 chunk 混在一起
MILVUS_COLLECTION=phoebe_chunks_dev
# 要測 LightOnOCR 時（需啟動上面的 vllm）
OCR_BACKEND=vllm
VLLM_BASE_URL=http://localhost:8001/v1
```

`KB_INGEST_MODE` 維持預設的 `embedded`：本機版的 ingest 跑在 web 程序裡，不用另外啟動。

**3. 兩個終端機各跑一個指令：**

```
uv run uvicorn app.main:create_app --factory --reload
cd frontend && npm run dev
```

開啟 http://localhost:5173。改後端程式，uvicorn 會自動重啟；改前端程式，畫面會立即更新。前端的 `/api` 請求會轉送到 8000 埠的後端。

**注意**

- 本機版的資料在專案根目錄的 `data/`（SQLite 與上傳檔），和 Docker 版的資料完全分開，需要另外註冊帳號。
- LiteParse 在本機（Windows）也能直接使用，`uv sync` 會一併安裝。
- 開發完要回到 Docker 版時：`cd docker && docker compose up -d`（要用 LightOnOCR 則加 `--profile lightonocr`）。

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
- **圖檔**：一張圖一個 chunk，視為單頁，依知識庫的解析器處理。

### Embedding 服務（embedder）

文字與圖片共用同一個 embedding 模型 `google/embeddinggemma-2`（768 維），由獨立的 `embedder` 服務提供（`embedder/`，sentence-transformers，CPU float32，不需 GPU）。應用程式以 `EMBEDDER_URL` 連線（本機預設 `http://localhost:8002`；Docker 內固定為 `http://embedder:8000`）。因此「用文字問、命中圖片內容」可以直接運作，不需要先把圖片轉成文字。

- 首次啟動會從 Hugging Face 下載模型（約 1–2 GB），第一次啟動會等很久；載入完成前 `GET /health` 回 503，檢索與解析會顯示「無法連線到 embedding 服務」。
- 本機啟動：`cd docker && docker compose up -d embedder`。細節與疑難排解見 [docker/README.md](docker/README.md)。

PDF 與圖檔的解析器是**知識庫層級的設定**（知識庫頁上方的「知識庫設定」，新建的知識庫預設 LiteParse），之後上傳的 PDF 與圖檔一律使用這個設定；txt、md 一律用文字解析，不受影響。

> **注意：改了解析器只影響之後上傳的檔案。** 已經上傳的文件會維持原本的解析器，不會自動重新處理，所以同一個知識庫裡可能同時有用不同解析器處理的文件（文件列表的「解析器」欄會顯示每份文件實際用的解析器）。想讓舊文件也改用新解析器，請刪除後重新上傳。
>
> 從沒有這個設定的舊版升級時，每個知識庫會依現有文件補上設定：PDF 與圖檔都用同一個解析器的，就沿用那個解析器；沒有這類文件或混用的，設為 LiteParse。

解析器的差異如下。每個解析器怎麼切 chunk、送什麼給 embedding 模型、回答時附哪些圖，詳見 [docs/parsers.md](docs/parsers.md)；實測的處理時間見 [docs/parser-benchmarks.md](docs/parser-benchmarks.md)。

| 解析器 | 適合 | 說明 |
| --- | --- | --- |
| LiteParse | 文字型 PDF | 預設、速度快、不需 GPU；掃描頁會用內建 Tesseract（`chi_tra+eng`）辨識，但中文品質有限。每頁另存一張頁面圖（400 DPI 渲染後縮到最長邊 1536 px，`page_{n}.png`），向量由「整頁圖＋該頁文字」產生（**頁面圖向量**），圖表、照片與版面也能被檢索到 |
| LightOnOCR-3-0.8B | 掃描檔、表格、複雜版面 | 在本機 GPU 執行，輸出 Markdown 與各區塊座標（grounding）；RTX 2060 上每頁約 24 秒（一次送 4 頁）。依座標把頁面中的圖片**裁切**成獨立的圖（`img_{頁}_{序}.png`，捨棄頁首、頁尾與頁碼）；向量**只由**頁面的 Markdown 文字產生（裁切圖本身不轉成向量，只靠模型寫的圖片描述被檢索到），裁切圖附在該頁的引用卡片上，也會一併附給聊天模型 |

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

指令會把所有「完成」與「失敗」的文件設為排隊中（清除錯誤與 chunk 數），由 ingest 依原解析器重新處理；正在處理中的文件會略過，待其完成後再執行一次即可。

## 測試

自動化測試分成三組，各自在對應的目錄執行。大部分測試把 LLM、embedder、Milvus、解析器換成測試替身，不需要 GPU、網路或 Docker。

**後端**（專案根目錄，約 3–4 分鐘）：

```
uv run pytest -q
```

- `-q` 只印每個測試的結果符號和最後的總結；`-v` 會逐一列出每個測試的名稱。
- 只跑某個檔案或某個測試：`uv run pytest tests/test_monitor.py`、`uv run pytest tests/test_monitor.py::test_chat_trace_finishes_once`。
- 連真實服務的整合測試也在裡面：標記為 `milvus`（需要 Milvus）與 `embedder`（需要 embedder 服務）的測試，服務連得上時會一起執行，連不上就自動略過。只跑或排除它們：

  ```
  uv run pytest -m milvus
  uv run pytest -m embedder
  uv run pytest -m "not milvus and not embedder"
  ```

  服務的位址用 `MILVUS_URI`、`EMBEDDER_URL` 指定（預設 `http://localhost:19530`、`http://localhost:8002`）；以 Docker 啟動：`cd docker && docker compose up -d milvus embedder`。

**embedder 服務**（它有自己的套件環境，用假模型，不下載模型，幾秒就跑完）：

```
cd embedder
uv run pytest -q
```

**前端**（約 1 分鐘）：

```
cd frontend
npx vitest run
npx tsc -b       # 型別檢查
npm run lint     # 程式風格檢查
```

**效能測試**（僅在需要時手動執行，不屬於上面的自動化測試；需要 Docker 服務都在跑，會跑好幾分鐘）：

```
uv run python -m scripts.benchmark
```

用固定的情境實際跑文件處理、搜尋、對話，報告寫到 `docs/benchmarks/`，並和基準比較，說明見 [docs/benchmarks.md](docs/benchmarks.md)。

**AI 品質評測**（僅手動執行，用 DeepEval；裁判模型由 `EVAL_JUDGE_MODEL` 決定，預設是 Gemini，付費方案會計費，免費方案每分鐘只能呼叫 15 次，每天 500 次，所以 RAG 評測要跑 1 小時左右，一天最好只跑一次；分數僅供觀察，不是通過／失敗的門檻）：

```
uv run --group eval pytest evals/test_rag_quality.py   # 知識庫問答（RAG），需要 Docker 服務都在跑
uv run --group eval pytest evals/test_chat_quality.py  # 一般聊天，需要本機 Ollama 的 CHAT_MODEL
```

RAG 評測用 `test_data/rag/` 的 25 份文件和 43 題，分開評「檢索有沒有找對段落」和「回答是否正確、有沒有編造」，報告寫到 `evals/reports/`，說明見 [docs/rag-eval.md](docs/rag-eval.md)。

注意：單純執行 `uv sync` 會移除 eval、ocr 群組的套件；`uv run --group eval`／`uv run --group ocr` 會重新安裝。

## 設定（.env）

設定可放在專案根目錄的 `.env`（本機快速測試用，範本見 `.env.example`；Docker 部署改用 `docker/.env`，範本見 `docker/.env.example`）。本機可用 `ENV_FILE` 指定其他路徑。同名的環境變數優先於 `.env`。`.env` 已被 git 忽略；測試不會讀取它。

## 管理員與系統設定

- 管理員：在 `.env`（Docker 為 `docker/.env`）設定 `ADMIN_USERNAME` 與 `ADMIN_PASSWORD`，服務啟動時會自動建立這個帳號，不需要註冊。之後以 `.env` 為準：改了密碼重新啟動就會生效（該帳號原本的登入會被登出）；若已有人註冊了同名帳號，密碼會被改成 `.env` 的值並成為管理員。`.env` 已被 git 忽略，但密碼是明文，請保護好這個檔案。
- 只有管理員會在側邊欄看到「系統設定」頁，所有使用者的聊天都套用這份設定：
  - LLM 供應商：本機 Ollama 或 Google Gemini（未設定 `GEMINI_API_KEY` 時 Gemini 選項停用）
  - 模型：從目前提供者即時取得模型清單後選擇
  - 系統提示詞：最多 4000 字
  - temperature：可自訂，或使用模型預設值
- Embedding（知識庫）一律使用獨立的 `embedder` 服務（`EMBEDDER_URL`），不受 LLM 供應商影響。
- 管理員另有「監控」頁：事件紀錄（文件處理、AI 對話、搜尋測試各階段的耗時）、資源使用（各容器與 GPU 的用量）、圖表（記憶體、CPU、GPU 隨時間的變化，並標出各任務的時間）。資源用量由 Docker 的 `monitor` 服務取樣，說明見 [docs/monitoring.md](docs/monitoring.md)。

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
| `LLM_PROVIDER` | `gemini` | 預設的聊天供應商（`gemini` 或 `ollama`），可在「系統設定」頁覆寫 |
| `OLLAMA_NUM_CTX` | `16384` | Ollama 每次對話的 context 長度。Ollama 本身預設只有 4096，知識庫回答帶整頁圖（每張約 740 token）時會被截斷 |
| `ADMIN_USERNAME` | 空 | 管理員帳號，啟動時自動建立（3–32 字元） |
| `ADMIN_PASSWORD` | 空 | 管理員密碼，每次啟動同步到資料庫（8 字元以上，最多 72 位元組） |
| `EVAL_JUDGE_MODEL` | `gemini-3.5-flash-lite` | 評測（`evals/`）的裁判模型：`gemini-` 開頭用 Gemini（需要 `GEMINI_API_KEY`），其他名稱用本機 Ollama 的模型。只有手動執行評測時使用，容器不讀取 |
| `DB_PATH` | `<專案根目錄>/data/app.db` | SQLite 資料庫路徑 |
| `SESSION_DAYS` | `7` | 登入工作階段有效天數 |
| `CHAT_THINK` | `true` | 是否讓模型輸出思考過程（`false`、`0`、`no`、`off` 為關閉，不分大小寫） |
| `EMBEDDER_URL` | `http://localhost:8002` | Embedding 服務（`embedder`）位址；Docker 內固定為 `http://embedder:8000`；更換模型後需重建知識庫 |
| `KB_DIR` | `<專案根目錄>/data/kb` | 上傳檔案與解析出的圖片存放位置 |
| `KB_TOP_K` | `5` | 每次回答檢索的段落數 |
| `KB_VECTOR_WEIGHT` | `0.5` | 混合查詢中向量的預設權重（0–1），關鍵字權重為 1 減去它，分數依 Milvus WeightedRanker 的方式正規化後加權相加。每個知識庫可以在「知識庫設定」自訂自己的權重（AI 對話使用它），沒有自訂的知識庫用這個預設值；知識庫頁的檢索測試可以試用不同權重，再按「設為這個知識庫的權重」儲存 |
| `KB_MAX_IMAGES` | `4` | 回答時附給模型的參考圖片上限（依參考資料順序，頁面圖優先；最長邊超過 1536 px 會先縮小）；`0` 不附圖。模型不支援圖片時會自動改為不帶圖重試一次 |
| `MAX_UPLOAD_MB` | `50` | 單一上傳檔案上限 |
| `OCR_MODEL` | `lightonai/LightOnOCR-3-0.8B` | LightOnOCR 模型（Hugging Face） |
| `LITEPARSE_OCR_LANGUAGE` | `chi_tra+eng` | LiteParse 內建 Tesseract 的語言 |
| `KB_INGEST_MODE` | `embedded` | 文件處理（ingest）：`embedded`（內嵌於 web）或 `external`（獨立 `python -m app.kb.ingest_main`） |
| `KB_INGEST_POLL_SECONDS` | `2` | ingest 輪詢間隔 |
| `INGEST_CONCURRENCY` | `1` | ingest 同時處理幾份文件 |
| `OCR_CONCURRENCY` | `4` | LightOnOCR 處理一份文件時，同時送給 vLLM 的頁數。`INGEST_CONCURRENCY` × `OCR_CONCURRENCY` 不要超過 vLLM 的 `--max-num-seqs`（預設 4） |
| `OCR_TIMEOUT_SECONDS` | `600` | 每一頁 OCR 請求的逾時秒數 |
| `KB_STALE_SECONDS` | `120` | 解析中逾時多久視為卡住並重新排入 |
| `KB_HEARTBEAT_SECONDS` | `30` | 處理中的文件刷新 `updated_at` 的間隔 |
| `REPLY_LOCK_TTL_SECONDS` | `600` | 對話回覆鎖逾時秒數 |
| `OCR_BACKEND` | `transformers` | `transformers` 或 `vllm` |
| `VLLM_BASE_URL` | `http://localhost:8001/v1` | vLLM 位址 |
| `OCR_VLLM_KV_CACHE_MEMORY` | `512M` | vLLM 的 KV cache 大小（僅 Docker，見 [docs/gpu-memory.md](docs/gpu-memory.md)） |
| `OCR_VLLM_GPU_UTIL` | `0.7` | vLLM 啟動時的顯存檢查比例，剩餘顯存不足就拒絕啟動（僅 Docker） |
| `OCR_VLLM_MAX_NUM_SEQS` | `4` | vLLM 同時處理的請求上限（僅 Docker） |
| `MILVUS_URI` | `http://localhost:19530` | Milvus 位址（快速測試：`cd docker && docker compose up -d milvus`；Docker 內固定為 `http://milvus:19530`） |
| `MILVUS_TOKEN` | 空 | Milvus 驗證 token（未啟用驗證則留空） |
| `MILVUS_COLLECTION` | `phoebe_chunks` | 存放知識庫段落的 collection 名稱 |
| `WEB_CONCURRENCY` | `2` | Gunicorn worker 數（僅 Docker，設在 `docker/.env`） |
| `PHOEBE_PORT` | `8000` | Docker 對外連接埠 |
| `MONITOR_INTERVAL_SECONDS` | `5` | 監控頁面的資源取樣間隔（秒，monitor 服務） |
| `MONITOR_RESOURCE_RETENTION_HOURS` | `48` | 資源取樣保留幾小時 |
| `MONITOR_EVENT_RETENTION_DAYS` | `30` | 任務事件保留幾天 |
| `FRONTEND_DIST` | `<專案根目錄>/frontend/dist` | 前端建置輸出資料夾，由後端提供；使用相對路徑時，會以啟動 uvicorn 的工作目錄為基準 |

## 部署

以 Docker Compose 部署（Gunicorn web、獨立 ingest 服務、embedder、Milvus；vLLM 為選用），詳見 [docker/README.md](docker/README.md)；各服務的功能與資料流程見 [docs/services.md](docs/services.md)：

```
cd docker
cp .env.example .env   # Docker 專用設定，與根目錄 .env 無關
docker compose up -d --build
```

vLLM（LightOnOCR）是選用的，不加 profile 不會啟動。要使用就帶上 `lightonocr` profile，停止時也要帶：

```
docker compose --profile lightonocr up -d --build
docker compose --profile lightonocr down
```

## 資料位置

資料存放在 SQLite 檔案 `data/app.db`，知識庫檔案在 `data/kb/`（位於專案根目錄，與啟動目錄無關；可用 `DB_PATH` 覆寫）。`data/` 已被 git 忽略。向量與 BM25 索引存放在 Milvus。Docker 部署時的資料位置見 [docs/services.md](docs/services.md#資料存在哪裡)。從舊版升級的方式見 [docker/README.md](docker/README.md)。

## 專案結構

- `app/` — 後端（設定、資料庫、驗證、對話、Ollama 客戶端、路由）
- `app/kb/` — 知識庫（資料存取、解析器、索引、背景處理）
- `app/monitor/` — 監控（任務事件、資源取樣、monitor 服務）
- `scripts/` — 手動驗證腳本
- `frontend/` — 前端（React + TypeScript + Vite）
- `tests/` — 單元測試
- `evals/` — AI 品質評測（DeepEval）：一般聊天、知識庫問答（RAG）

## 開發流程

所有變更都透過 feature 分支與 Pull Request 進行，由 repo 擁有者審查後合併，不直接推送到 `main`。

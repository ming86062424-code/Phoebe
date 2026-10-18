# 部署基礎（Docker、Gunicorn、.env）— 設計規格

- 日期：2026-10-18
- 子專案：1 / 4（部署基礎 → LLM 供應商切換 → Milvus 混合查詢 → 多模態知識庫）
- 狀態：使用者授權直接開發（規格與計畫不需逐段確認）

## 1. 目的與範圍

**目的**：讓 Phoebe 能以兩種方式啟動，並為後續子專案打好設定與部署基礎：
- **快速測試**：在 Windows 上 `uv run uvicorn …` 直接啟動，不需要 Docker
- **正式部署**：在這台 Windows 的 Docker Desktop（WSL2 + NVIDIA GPU）以 `docker compose` 啟動，web 用 Gunicorn + Uvicorn worker

**需求（使用者確認）**
1. Gunicorn + Uvicorn 部署
2. 根目錄 `docker/` 存放所有 Docker 相關檔案
3. 快速測試與 Docker 正式部署都要能用
4. OCR 之後可能改用 vLLM；Windows 無法跑 vLLM，**快速測試不能被 vLLM 卡住**
5. 環境變數集中在 `.env`，並提供 `.env.example`
6. Docker 部署時一律連主機上的 Ollama（不在 compose 裡跑 Ollama）
7. 正式部署目標就是這台 Windows（Docker Desktop），GPU 為 RTX 2060 6GB

**明確不做（YAGNI）**：Kubernetes、HTTPS／反向代理、CI/CD、PostgreSQL、Ollama 容器、Milvus（子專案 3）、Gemini（子專案 2）。

## 2. 現況與要解決的問題

| 現況 | 多 process（Gunicorn 多 worker + 獨立解析 worker）時的問題 |
|---|---|
| 知識庫解析 worker 是 web process 內的執行緒，佇列在記憶體 | 每個 Gunicorn worker 各啟動一個，重複處理、OCR 模型重複載入 |
| 回覆並發限制用 `app.state.replying`（記憶體 set） | 只在單一 process 內有效 |
| 向量庫用 Chroma `PersistentClient`（本機檔案） | 多 process 同時寫入同一資料夾不安全 |
| SQLite 預設 journal 模式 | 多 process 寫入容易 `database is locked` |
| 設定只讀 `os.environ` | 沒有 `.env` |

## 3. 設計

### 3.1 設定與 `.env`
- 新增依賴 `python-dotenv`。`Settings.from_env()` 先以 `load_dotenv(PROJECT_ROOT / ".env", override=False)` 載入（已存在的環境變數優先，方便 Docker 與測試覆寫）
- `ENV_FILE` 環境變數可指定其他檔案路徑
- 新增 `.env.example`（提交進 git）：列出所有設定、預設值與一行中文說明，分區塊（伺服器、Ollama、資料、知識庫、OCR、部署）
- `.env` 不進 git（`.gitignore` 已含）
- 測試不得讀到開發者本機的 `.env`：`tests/conftest.py` 設 `ENV_FILE` 指向不存在的檔案

**新增設定**

| 變數 | 預設 | 說明 |
|---|---|---|
| `KB_WORKER_MODE` | `embedded` | `embedded`：web process 內啟動解析 worker；`external`：不啟動，由獨立 worker 服務處理 |
| `OCR_BACKEND` | `transformers` | `transformers`：本機載入模型（需 `--group ocr`）；`vllm`：呼叫 vLLM 的 OpenAI 相容 API |
| `VLLM_BASE_URL` | `http://localhost:8001/v1` | `OCR_BACKEND=vllm` 時使用 |
| `CHROMA_HOST` | （空） | 空：本機檔案模式（`CHROMA_DIR`）；有值：連 Chroma 伺服器（`CHROMA_PORT`，預設 8000） |
| `CHROMA_PORT` | `8000` | |
| `REPLY_LOCK_TTL_SECONDS` | `90` | 回覆鎖逾時（process 異常結束時避免永久鎖住）；串流中每 30 秒更新鎖，故需大於 30 |

### 3.2 解析佇列改為 DB 驅動
- `kb_documents.status = 'queued'` 即為佇列；worker 以單一 SQL 原子性地「認領」一份：`UPDATE kb_documents SET status='parsing', updated_at=? WHERE id = (SELECT id FROM kb_documents WHERE status='queued' ORDER BY id LIMIT 1) AND status='queued' RETURNING id`（SQLite 3.35+ 支援 `RETURNING`）
- worker 迴圈：認領 → 處理 → 再認領；沒有工作時睡 `KB_WORKER_POLL_SECONDS`（預設 2 秒），`enqueue()` 可喚醒同 process 的 worker（embedded 模式維持即時性）
- 逾時重排：把 `parsing`／`indexing` 但 `updated_at` 超過 `KB_STALE_SECONDS`（預設 120）的文件重設為 `queued`（取代現在「啟動時全部重排」的行為，避免多個 worker 互搶）；worker 啟動時執行一次，閒置時約每 60 秒再執行一次（與孤立檔案掃描同頻率）
- 心跳：處理文件期間（解析與建立索引兩階段）有背景執行緒每 `KB_HEARTBEAT_SECONDS`（預設 30）刷新該文件的 `updated_at`，不論有無進度，因此長時間處理中的文件不會被誤判為逾時
- 上傳時文件先以 `pending`（不可被認領）建立，檔案就位後才改為 `queued`
- 現有的文件身分檢查（PR #3 review #1 的修正）保留
- 新增獨立入口 `python -m app.kb.worker_main`：建立 `KbServices`、執行 worker 迴圈、處理 SIGTERM 優雅結束
- `KB_WORKER_MODE=embedded`（預設）：維持現在的行為（lifespan 啟動 worker）；`external`：lifespan 不啟動
- 解析器可用性：web 容器沒有 torch 與 vLLM，無法自行判斷。新資料表 `parser_status (parser_id, available, reason, updated_at)`；worker 以獨立的 daemon 執行緒每 15 秒呼叫各解析器的 `availability()` 並寫入（不放在 `run_forever` 的閒置維護，免得長時間 OCR 讓狀態過期）。`KbServices.availability()`：external 模式讀此表，無資料或 `updated_at` 超過 45 秒 → `(False, "文件解析服務（worker）未啟動")`；embedded 模式維持本機檢查。`GET /api/parsers` 與上傳都經由它
- 週期維護另會：刪除超過 1 小時的 `upload-*.tmp`；把超過 `KB_STALE_SECONDS` 仍是 `pending` 的文件標為 `failed`（「上傳未完成，請重新上傳」）；`Settings.from_env` 要求 `KB_HEARTBEAT_SECONDS < KB_STALE_SECONDS`

### 3.3 回覆鎖改為 DB
- 新資料表 `active_replies (conversation_id INTEGER PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE, started_at TEXT NOT NULL)`
- `claim_reply`：先刪除該對話中超過 `REPLY_LOCK_TTL_SECONDS` 的舊鎖，再 `INSERT`；`IntegrityError` → 409（沿用現有訊息）。回傳的鎖物件可呼叫以釋放（刪除該列，冪等），並提供 `touch()`：串流期間 `body_iter` 每 30 秒刷新自己那一列的 `started_at`（心跳），所以 web 異常重啟後最多 `REPLY_LOCK_TTL_SECONDS` 秒鎖即失效。async 處理常式以 `run_in_threadpool` 呼叫 `claim_reply`
- 移除 `app.state.replying`

### 3.4 SQLite 多 process
- `connect()` 加上 `PRAGMA journal_mode=WAL`、`PRAGMA busy_timeout=5000`
- Docker 一律用 named volume 存 `data/`（不用 Windows bind mount，WSL2 的檔案鎖在 bind mount 上不可靠）

### 3.5 Chroma 伺服器模式（過渡，子專案 3 移除）
- `KnowledgeIndex` 依設定使用 `chromadb.PersistentClient(path=…)` 或 `chromadb.HttpClient(host, port)`
- compose 內放一個 `chroma` 服務（官方映像，固定版本），named volume 持久化

### 3.6 OCR 後端
- `LightOnOcrParser` 拆成「把 PDF 轉成頁面影像」與「對一張影像做 OCR」兩層；OCR 由 `OcrEngine` 協定提供：`ocr(image) -> str`、`availability() -> tuple[bool, str]`、`release()`
  - `TransformersOcrEngine`：現有的本機 transformers 實作（含卸載 Ollama 模型、fp16／bf16 判斷）
  - `VllmOcrEngine`：以 httpx 呼叫 `{VLLM_BASE_URL}/chat/completions`，影像以 base64 PNG data URL 送出，只送頁面影像、`temperature=0.1`、`chat_template_kwargs={"enable_thinking": false}`、`max_tokens=4096`（依 LightOnOCR 官方 README：不要用貪婪解碼；transformers 引擎同樣 `do_sample=True, temperature=0.1`，故重跑結果可能略有差異）。兩個引擎的輸出都會經 `fix_escaped_dollars()`，把成對的跳脫數學分隔符號還原成 `$…$`，單獨出現的金額不動，模型名用 `OCR_MODEL`；`availability()` 以 `GET {VLLM_BASE_URL}/models` 檢查，連不上時回「OCR 服務（vLLM）未啟動」
- Docker 預設 `OCR_BACKEND=vllm`（worker 於 compose 固定）：GPU 專用於 OCR，vllm 以官方 0.8B 參數啟動（`--trust-remote-code --limit-mm-per-prompt '{"image": 1}' --default-chat-template-kwargs '{"enable_thinking": false}' --mm-processor-cache-gb 0 --no-enable-prefix-caching --max-model-len 18000`），Turing 加 `--dtype float16`；頁面 400 DPI 算繪、縮圖到最長邊 2048。Ollama 卸載模型只在 transformers 引擎（本機快速測試）保留
- `OCR_BACKEND` 決定使用哪個；transformers／torch 只在 `transformers` 引擎被用到時才 import，所以 `vllm` 模式的映像不需要 torch
- 文件標示的 parser id 不變（仍為 `lightonocr`），使用者介面不需改

### 3.7 Gunicorn
- 新增依賴 `gunicorn`、`uvicorn-worker`（只在 Linux 容器內使用；Windows 不執行 gunicorn）
- 設定檔 `docker/gunicorn.conf.py`：`bind = 0.0.0.0:8000`、`worker_class = "uvicorn_worker.UvicornWorker"`、`workers` 讀 `WEB_CONCURRENCY`（預設 2）、`timeout = 120`、`graceful_timeout = 30`、`keepalive = 5`、access log 到 stdout
- `on_starting` hook 在 master 於 fork 前執行一次 `init_db`（不使用 preload_app）。`init_db` 以 `BEGIN IMMEDIATE` 交易包住全部遷移（逾時 30 秒），可跨程序並行執行；重建 AUTOINCREMENT 表時保留 `sqlite_sequence`。
- 啟動指令：`gunicorn -c docker/gunicorn.conf.py "app.main:create_app()"`；容器內 `KB_WORKER_MODE=external`

### 3.8 `docker/` 目錄

```
docker/
├── Dockerfile            # 多階段：frontend（node 24 建置）→ web（python 3.12 slim + uv，無 torch）→ worker（加上 ocr 依賴與 CUDA 版 torch）
├── docker-compose.yml           # 正式部署：web、worker、chroma；vllm 為預設服務
├── gunicorn.conf.py
├── .dockerignore 的內容放在根目錄 .dockerignore（Docker 規定位置；compose 以根目錄為 build context）
└── README.md             # 部署說明（中文）
```

**compose 服務**

| 服務 | 映像／target | 說明 |
|---|---|---|
| `web` | Dockerfile target `web` | Gunicorn；port `${PHOEBE_PORT:-8000}:8000`；`KB_WORKER_MODE=external`；`CHROMA_HOST=chroma`；`OLLAMA_BASE_URL=http://host.docker.internal:11434`；`DB_PATH`、`KB_DIR` 指向 `/data` volume |
| `worker` | target `worker` | `python -m app.kb.worker_main`；同樣的 `/data` volume；`OCR_BACKEND=vllm`、`VLLM_BASE_URL=http://vllm:8000/v1`（固定）；不含 torch、不使用 GPU；等 `vllm` healthy 才啟動 |
| `chroma` | `chromadb/chroma:<固定版本>` | volume `chroma-data` |
| `vllm`（預設服務） | `vllm/vllm-openai:v0.30.0` | 官方 0.8B 參數（見 §3.6）、`--dtype float16`、`--gpu-memory-utilization ${OCR_VLLM_GPU_UTIL:-0.8}`、`--max-model-len ${OCR_VLLM_MAX_MODEL_LEN:-18000}`、`--enforce-eager`、`--max-num-seqs ${OCR_VLLM_MAX_NUM_SEQS:-8}`（RTX 2060 實測：0.85 失敗、0.8 但使用 CUDA graph 與預設 256 條序列配不到 KV cache；上述組合 KV cache 1.28 GiB／99,243 tokens）；不載入 env_file（避免未知 `VLLM_*` 警告）；需要 NVIDIA 驅動 ≥ 580（CUDA 13 映像）；port `8001:8000`；使用 GPU；`hf-cache` 掛在 `/root/.cache/huggingface`；healthcheck `GET /health`，`start_period` 600 秒 |

- 所有服務都讀 `env_file: [{path: .env, required: false}]`（即 `docker/.env`），再以 `environment:` 覆寫容器內必須不同的值（路徑、主機名稱）
- Docker 專用設定檔 `docker/.env`（範本 `docker/.env.example`，已被 git 忽略並排除於 build context）；指令在 `docker/` 目錄執行，Compose 自動讀取它做變數插值，`env_file: [{path: .env, required: false}]` 供容器環境變數。根目錄 `.env` 只給本機快速測試。compose `environment:` 固定的鍵（DB_PATH、KB_DIR、CHROMA_HOST/PORT、OLLAMA_BASE_URL、KB_WORKER_MODE、VLLM_BASE_URL、HF_HOME）不可由 `.env` 覆寫
- 啟動順序：`chroma`（healthcheck）→ `web`（healthcheck：`GET /api/me` 回 401 視為健康）→ `worker`；`web`、`worker` 設 `stop_grace_period: 30s`；worker 固定覆寫 `VLLM_BASE_URL=http://vllm:8000/v1`
- 啟動：`cd docker && docker compose up -d --build`；vLLM 為預設服務，worker 的 `OCR_BACKEND`、`VLLM_BASE_URL` 由 compose 固定
- 主機 Ollama：`host.docker.internal`；README 說明若連不上需讓 Ollama 監聽 `0.0.0.0`（`OLLAMA_HOST`）
- **GPU 專用於 OCR**：對話模型可放雲端，不替 Ollama 保留顯存；RTX 2060（6GB，Turing 無 bf16）以 fp16、`OCR_VLLM_GPU_UTIL` 0.8 執行

### 3.9 快速測試
- `cp .env.example .env` → 依需要修改 → `uv sync --group ocr`（要用 OCR 時）→ `uv run uvicorn app.main:create_app --factory --reload`
- 預設 `KB_WORKER_MODE=embedded`、`OCR_BACKEND=transformers`、Chroma 本機模式 → Windows 上不需要 Docker、也不會碰到 vLLM
- 也可只用 Docker 跑 vLLM（`cd docker && docker compose up -d vllm`）並在本機 `.env` 設 `OCR_BACKEND=vllm`、`VLLM_BASE_URL=http://localhost:8001/v1`（compose 把 vllm 的 8000 對應到主機 8001）

## 4. 錯誤處理
- `OCR_BACKEND` 值不合法 → 啟動時拋出清楚的錯誤訊息
- vLLM 連不上：該 parser `availability()` 回 False 與說明（上傳介面現有機制會顯示）；處理中途失敗 → 文件標 `failed`，錯誤訊息「OCR 服務（vLLM）呼叫失敗：…」
- worker 收到 SIGTERM：完成目前這一頁後停止，未完成的文件保持 `parsing`，下次啟動時若逾時會重新排入

## 5. 測試
- `.env` 載入：`.env` 檔的值會被讀到；環境變數優先於 `.env`；`ENV_FILE` 生效
- 佇列認領：兩個 worker（兩條連線）同時認領 → 同一份文件只被一個認領；逾時的 `parsing` 會被重排、未逾時的不會
- 回覆鎖：兩條連線同時 claim 同一對話 → 第二個 409；release 後可再 claim；逾時的鎖會被清掉
- `KB_WORKER_MODE=external` 時 lifespan 不啟動 worker
- `VllmOcrEngine`：用 `httpx.MockTransport` 驗證請求格式（data URL、temperature 0、model）、回應解析、連線失敗訊息、`availability()`
- `OcrEngine` 選擇：`OCR_BACKEND` 切換；不合法值報錯
- Chroma：`CHROMA_HOST` 有值時建立 HttpClient（以替身驗證，不連真的伺服器）
- WAL：`connect()` 後 `PRAGMA journal_mode` 為 `wal`
- Docker（實作最後一步，需 Docker Desktop 執行中）：`docker compose build` 成功；`up` 後 `GET /` 回前端、`/api/me` 回 401；註冊、登入、送一則訊息（主機 Ollama）成功；上傳一份小 PDF 用 LiteParse 解析到 `ready`
- 既有的後端 pytest、前端 vitest、lint、typecheck、build 全部通過

## 6. 開發流程
- 分支 `feature/deployment`（從 `main`），完成後開 PR

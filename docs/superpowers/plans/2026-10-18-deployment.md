# 部署基礎（Docker、Gunicorn、.env）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phoebe 可以在 Windows 上不用 Docker 快速啟動，也能用 Docker Desktop 以 Gunicorn 多 worker + 獨立解析 worker 正式部署；設定集中在 `.env`。

**Architecture:** 設定改由 `.env` + 環境變數載入；所有跨 process 的協調（解析佇列、回覆鎖）改放 SQLite（WAL）；解析 worker 可以內嵌在 web process 或獨立執行；OCR 引擎可在本機 transformers 與 vLLM HTTP 之間切換；Chroma 支援伺服器模式；`docker/` 提供多階段 Dockerfile、compose 與 Gunicorn 設定。

**Tech Stack:** Python 3.12、FastAPI、python-dotenv、gunicorn + uvicorn-worker、SQLite WAL、chromadb（PersistentClient / HttpClient）、httpx、Docker Desktop（WSL2 + NVIDIA）、Docker Compose v2

**Spec:** `docs/superpowers/specs/2026-10-18-deployment-design.md`

## Global Constraints

- 新設定與預設值：`KB_WORKER_MODE=embedded`（`embedded|external`）、`OCR_BACKEND=transformers`（`transformers|vllm`）、`VLLM_BASE_URL=http://localhost:8001/v1`、`CHROMA_HOST=`（空）、`CHROMA_PORT=8000`、`REPLY_LOCK_TTL_SECONDS=600`、`KB_WORKER_POLL_SECONDS=2`、`KB_STALE_SECONDS=120`、`KB_HEARTBEAT_SECONDS=30`（處理中每 30 秒刷新 `updated_at`；閒置 worker 約每 60 秒重排逾時文件）、`WEB_CONCURRENCY=2`（只有 gunicorn 用）
- `KB_WORKER_MODE`、`OCR_BACKEND` 值不合法 → `Settings.from_env()` 拋 `ValueError`，訊息為中文並列出合法值
- `.env` 載入：`load_dotenv(path, override=False)`；路徑為 `ENV_FILE` 或 `<專案根>/.env`；環境變數優先
- 測試永遠不讀開發者的 `.env`（conftest 把 `ENV_FILE` 設成不存在的路徑）
- SQLite 連線：`PRAGMA foreign_keys = ON`、`PRAGMA journal_mode=WAL`、`PRAGMA busy_timeout=5000`
- 回覆鎖逾時、解析逾時都以 `updated_at`／`started_at` 的 UTC ISO 字串比較（沿用 `now_iso()`）
- 409 回覆鎖訊息沿用現有文字（不改）
- Gunicorn 只在 Linux 容器內使用；Windows 快速測試一律 `uv run uvicorn app.main:create_app --factory`
- Docker：所有資料用 named volume（不用 bind mount）；`env_file` 指向根目錄 `.env` 且 `required: false`；主機 Ollama 為 `http://host.docker.internal:11434`
- 映像檔版本一律固定（不得用 `latest`）；Docker 指令以 `docker compose -f docker/compose.yml …` 執行，build context 為專案根目錄
- 介面與文件文字為繁體中文
- Windows、Git Bash；`uv` 在 `/c/Users/anny3/AppData/Roaming/Python/Python312/Scripts`；Node 在 `/c/Program Files/nodejs`；`PYTHONUTF8=1`；LF 換行
- 每個 commit 訊息結尾：空一行後 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

1. **開發者本機的 `.env` 影響測試結果**：期望測試完全不受 `.env` 影響。→ Task 1 `test_tests_ignore_developer_env_file`
2. **兩個 worker 同時搶同一份文件**：期望只有一個認領成功。→ Task 3 `test_two_connections_claim_distinct_documents`
3. **worker 當掉，文件永遠停在「解析中」**：期望逾時後重新排入，未逾時的不受影響。→ Task 3 `test_stale_parsing_documents_are_requeued`
4. **web process 當掉，對話被永久鎖住無法再回覆**：期望鎖逾時後自動失效。→ Task 2 `test_expired_reply_lock_is_replaced`
5. **external 模式下刪不掉的檔案從此沒人清理**：期望獨立 worker 會清掉沒有對應文件的檔案。→ Task 3 `test_cleanup_orphans_removes_files_without_documents`

---

## 檔案結構

| 檔案 | 職責 |
|---|---|
| `app/config.py` | `.env` 載入、新設定、驗證 |
| `.env.example` | 所有設定的範例與說明 |
| `app/db.py` | WAL、busy_timeout、`active_replies` 表 |
| `app/chat_stream.py`、`app/routes/chat.py`、`app/main.py` | DB 回覆鎖；依 `KB_WORKER_MODE` 啟動 worker |
| `app/kb/store.py` | `claim_next_document`、`requeue_stale_documents` |
| `app/kb/worker.py` | DB 驅動的處理迴圈、孤兒檔案清理 |
| `app/kb/worker_main.py` | 獨立 worker 入口 |
| `app/kb/parsers/ocr_engines.py` | `OcrEngine`、`TransformersOcrEngine`、`VllmOcrEngine`、`build_ocr_engine` |
| `app/kb/parsers/lightonocr.py` | 改為使用 `OcrEngine` |
| `app/kb/index.py`、`app/kb/services.py` | Chroma 用戶端選擇；組裝 |
| `docker/Dockerfile`、`docker/compose.yml`、`docker/gunicorn.conf.py`、`docker/README.md`、`.dockerignore` | 部署 |

---

### Task 1: `.env` 設定載入與 SQLite WAL

**Files:**
- Modify: `app/config.py`、`app/db.py`、`tests/conftest.py`、`pyproject.toml`（`uv add python-dotenv`）
- Create: `.env.example`
- Test: `tests/test_config.py`（新）、`tests/test_db.py`

**Interfaces:**
- Produces:
  - `app.config.env_file_path() -> Path`：`ENV_FILE` 或 `PROJECT_ROOT / ".env"`
  - `Settings.from_env()`：開頭呼叫 `load_dotenv(env_file_path(), override=False)`；新增欄位 `kb_worker_mode: str`、`ocr_backend: str`、`vllm_base_url: str`、`chroma_host: str`、`chroma_port: int`、`reply_lock_ttl_seconds: int`、`kb_worker_poll_seconds: float`、`kb_stale_seconds: int`（預設見 Global Constraints）
  - 驗證：不合法時 `ValueError("KB_WORKER_MODE 必須是 embedded 或 external")`、`ValueError("OCR_BACKEND 必須是 transformers 或 vllm")`
  - `connect()` 加上 WAL 與 busy_timeout
  - `.env.example`：涵蓋 `Settings.from_env()` 讀的每一個變數與 `WEB_CONCURRENCY`、`PHOEBE_PORT`、`VLLM_GPU_UTIL`，分區塊、每行附中文註解，值為預設值

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
def test_env_file_values_are_loaded(tmp_path, monkeypatch):      # 寫 CHAT_MODEL=from-file 到 tmp .env，ENV_FILE 指向它，delenv CHAT_MODEL → chat_model == "from-file"
def test_environment_overrides_env_file(tmp_path, monkeypatch):  # 檔案 from-file、環境 from-env → "from-env"
def test_tests_ignore_developer_env_file():                      # os.environ["ENV_FILE"] 指向不存在的檔案
def test_new_settings_defaults(monkeypatch):                     # 各新欄位等於預設值
def test_invalid_worker_mode_and_ocr_backend_raise(monkeypatch)  # 兩種都 pytest.raises(ValueError, match="KB_WORKER_MODE") / match="OCR_BACKEND"
def test_env_example_lists_every_setting():
    # 解析 .env.example 的 KEY= 行；Settings.from_env 原始碼中 os.environ.get("X" 的每個 X 都在其中
# tests/test_db.py
def test_connect_enables_wal_and_busy_timeout(tmp_path):         # journal_mode == "wal"；busy_timeout == 5000
```
注意：`load_dotenv` 會把值寫進 `os.environ`，測試要用 `monkeypatch` 在結束後清掉檔案帶進來的變數。

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py tests/test_db.py -q`
Expected: FAIL

- [ ] **Step 3: Implement**；`tests/conftest.py` 在模組最上方 `os.environ["ENV_FILE"] = str(Path(__file__).parent / "no-such.env")`

- [ ] **Step 4: Verify**

Run: `uv run pytest -q`
Expected: 全部通過

- [ ] **Step 5: Commit**

```bash
git add app/config.py app/db.py tests .env.example pyproject.toml uv.lock
git commit -m "feat: load settings from .env and enable SQLite WAL"
```

---

### Task 2: 回覆鎖改用資料庫

**Files:**
- Modify: `app/db.py`（新表）、`app/chat_stream.py`、`app/routes/chat.py`、`app/main.py`（移除 `app.state.replying`）
- Test: `tests/test_chat_concurrency.py`

**Interfaces:**
- Consumes: `Settings.reply_lock_ttl_seconds`（Task 1）
- Produces:
  - 資料表 `active_replies (conversation_id INTEGER PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE, started_at TEXT NOT NULL)`，由 `init_db` 建立
  - `app.chat_stream.claim_reply(db_path: str, conversation_id: int, ttl_seconds: int) -> Callable[[], None]`：自己開連線；刪除該對話 `started_at` 早於 `now - ttl` 的列，再 INSERT；`sqlite3.IntegrityError` → `HTTPException(409, <現有訊息>)`；回傳的 release 刪除該列且冪等

- [ ] **Step 1: Write the failing tests**

```python
def test_second_claim_on_same_conversation_is_409(settings, ...)   # 兩次 claim_reply（各自連線）→ 第二次 409
def test_release_allows_claiming_again(...)
def test_release_is_idempotent(...)
def test_expired_reply_lock_is_replaced(...)   # 手動插入 started_at 為 ttl+1 秒前的列 → claim 成功
def test_claims_on_different_conversations_do_not_conflict(...)
```
既有的並發測試改用新介面後必須照樣通過。

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chat_concurrency.py -q`
Expected: FAIL

- [ ] **Step 3: Implement**

- [ ] **Step 4: Verify**

Run: `uv run pytest -q`
Expected: 全部通過

- [ ] **Step 5: Commit**

```bash
git add app tests
git commit -m "feat: store reply locks in SQLite so they work across processes"
```

---

### Task 3: DB 驅動的解析佇列、獨立 worker、Chroma 伺服器模式

**Files:**
- Modify: `app/kb/store.py`、`app/kb/worker.py`、`app/kb/services.py`、`app/kb/index.py`、`app/main.py`
- Create: `app/kb/worker_main.py`
- Test: `tests/test_kb_store.py`、`tests/test_kb_worker.py`、`tests/test_kb_index.py`、`tests/test_kb_api.py`

**Interfaces:**
- Consumes: `Settings.kb_worker_mode / kb_worker_poll_seconds / kb_stale_seconds / chroma_host / chroma_port`（Task 1）
- Produces:
  - `store.claim_next_document(conn) -> int | None`：規格 §3.2 的單一 `UPDATE … RETURNING id`，把 `queued` 改成 `parsing` 並更新 `updated_at`
  - `store.requeue_stale_documents(conn, stale_before: str) -> list[int]`：`status IN ('parsing','indexing') AND updated_at < stale_before` → `queued`、`progress_done=0`
  - `IngestWorker.__init__` 新增 `poll_seconds: float`、`stale_seconds: int`、`kb_dir: str`
  - `IngestWorker.start()`：呼叫 `requeue_stale_documents`；`threaded=True` 時啟動執行緒跑 `run_forever`
  - `IngestWorker.run_forever(stop: threading.Event) -> None`：認領 → `process` → 再認領；沒有工作時等 `poll_seconds`（可被 `wake()` 提早叫醒），閒置時釋放 parsers、執行 `cleanup()` 與 `cleanup_orphans()`
  - `IngestWorker.wake() -> None`；`enqueue(doc_id)`：`threaded=True` 時 `wake()`；`threaded=False`（測試）時同步認領並處理到沒有工作為止
  - `IngestWorker.process(doc_id)` 前提改為「已被認領（status 為 parsing）」
  - `IngestWorker.cleanup_orphans() -> int`：掃描 `kb_dir` 下的文件檔與知識庫資料夾，刪除 DB 中已不存在的（回傳刪除數）；檔名規則沿用 `file_path(kb_id, doc_id)` 的現有格式
  - `app/kb/worker_main.py`：`main() -> None`——`Settings.from_env()`、`init_db`、`build_kb_services(settings)`、註冊 SIGTERM/SIGINT 設定 stop event、`worker.start()` 後在主執行緒跑 `run_forever`；`if __name__ == "__main__": main()`
  - `create_app`：`settings.kb_worker_mode == "embedded"` 才在 lifespan 啟動／停止 worker
  - `KnowledgeIndex.__init__(chroma_dir: str, embed_model, chroma_host: str = "", chroma_port: int = 8000)`：`chroma_host` 有值用 `chromadb.HttpClient(host=, port=, settings=…anonymized_telemetry=False)`，否則 `PersistentClient`

- [ ] **Step 1: Write the failing tests**

```python
# test_kb_store.py
def test_claim_next_document_claims_oldest_queued(...)        # 兩份 queued → 先拿到 id 小的，status 變 parsing
def test_two_connections_claim_distinct_documents(...)        # 兩條連線輪流認領 → 兩個不同 id；第三次 None
def test_stale_parsing_documents_are_requeued(...)            # updated_at 很舊的 parsing → queued；新的 parsing 不動
# test_kb_worker.py
def test_enqueue_unthreaded_processes_claimed_documents(...)  # 既有的處理流程測試改用新流程後仍通過
def test_run_forever_stops_on_event(...)                      # stop 已 set → 立即返回
def test_cleanup_orphans_removes_files_without_documents(...) # kb_dir 放一個不存在文件 id 的檔案 → 被刪；存在的保留
# test_kb_api.py
def test_external_worker_mode_does_not_start_worker(...)      # kb_worker_mode="external" 的 app，進入 lifespan 後 worker 執行緒未啟動
# test_kb_index.py
def test_chroma_host_uses_http_client(monkeypatch, ...)       # monkeypatch chromadb.HttpClient 為替身 → 以 host/port 呼叫
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_kb_store.py tests/test_kb_worker.py tests/test_kb_api.py tests/test_kb_index.py -q`
Expected: FAIL

- [ ] **Step 3: Implement**；`build_kb_services` 傳入新參數；web 端 `remove_path` 刪不掉時仍加入 pending（embedded 模式下由同 process 的 worker 重試），external 模式由 `cleanup_orphans` 補救

- [ ] **Step 4: Verify**

Run: `uv run pytest -q`
Expected: 全部通過

- [ ] **Step 5: Commit**

```bash
git add app tests
git commit -m "feat: DB-driven ingest queue, standalone worker and Chroma server mode"
```

---

### Task 4: OCR 引擎（transformers / vLLM）

**Files:**
- Create: `app/kb/parsers/ocr_engines.py`
- Modify: `app/kb/parsers/lightonocr.py`、`app/kb/services.py`
- Test: `tests/test_parsers.py`、`tests/test_ocr_engines.py`（新）

**Interfaces:**
- Consumes: `Settings.ocr_backend / vllm_base_url / ocr_model / ollama_base_url`
- Produces:
  - `OcrEngine`（Protocol）：`availability() -> tuple[bool, str]`、`ocr(image: "PIL.Image.Image") -> str`、`release() -> None`
  - `TransformersOcrEngine(model_id: str, ollama_base_url: str)`：現有 `_ensure_loaded`／`_ocr_image`／`release` 搬過來，行為不變；availability 訊息沿用
  - `VllmOcrEngine(base_url: str, model: str, transport: httpx.BaseTransport | None = None)`：
    - `availability()`：`GET {base_url}/models`，2xx → `(True, "")`，連線錯誤或非 2xx → `(False, "OCR 服務（vLLM）未啟動")`
    - `ocr(image)`：POST `{base_url}/chat/completions`，body `{"model": model, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,<…>"}}]}], "temperature": 0, "max_tokens": 4096}`；回傳 `choices[0].message.content`；HTTP 錯誤 → `ParseError(f"OCR 服務（vLLM）呼叫失敗：{detail}")`；逾時 300 秒
    - `release()`：不做事
  - `build_ocr_engine(settings: Settings) -> OcrEngine`
  - `LightOnOcrParser(engine: OcrEngine)`：`id`、`label` 不變；`availability()` 轉給 engine；`parse()` 負責 PDF → 頁面影像（現有 `RENDER_DPI`、`MAX_EDGE`）並呼叫 `engine.ocr`

- [ ] **Step 1: Write the failing tests**

```python
# test_ocr_engines.py（MockTransport）
def test_vllm_ocr_sends_png_data_url_and_parses_reply(...)   # 檢查 path、model、temperature==0、url 以 "data:image/png;base64," 開頭；回傳內容
def test_vllm_ocr_http_error_raises_parse_error(...)         # 500 → ParseError，訊息含「OCR 服務（vLLM）呼叫失敗」
def test_vllm_availability(...)                               # /models 200 → True；ConnectError → (False, "OCR 服務（vLLM）未啟動")
def test_build_ocr_engine_selects_backend(...)               # transformers / vllm 各回傳對應型別
# test_parsers.py
def test_lightonocr_parser_uses_engine_per_page(...)         # 假 engine 記錄呼叫次數 == PDF 頁數；文字對應頁碼
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ocr_engines.py tests/test_parsers.py -q`
Expected: FAIL

- [ ] **Step 3: Implement**（不得在模組頂層 import torch／transformers）

- [ ] **Step 4: Verify**

Run: `uv run pytest -q`
Expected: 全部通過

- [ ] **Step 5: Commit**

```bash
git add app tests
git commit -m "feat: pluggable OCR engine with transformers and vLLM backends"
```

---

### Task 5: Docker 與 Gunicorn

**前置（controller 處理）：** Docker Desktop 必須在執行中（`docker version` 能顯示 Server）。

**Files:**
- Create: `docker/Dockerfile`、`docker/compose.yml`、`docker/gunicorn.conf.py`、`docker/README.md`、`.dockerignore`
- Modify: `pyproject.toml`（`uv add gunicorn uvicorn-worker`）、`README.md`
- Test: `tests/test_gunicorn_conf.py`（新）

**Interfaces:**
- Consumes: 全部前面的設定；`python -m app.kb.worker_main`（Task 3）
- Produces:
  - `docker/gunicorn.conf.py`：規格 §3.7 的值；`workers = int(os.environ.get("WEB_CONCURRENCY", "2"))`
  - `docker/Dockerfile` 階段：`frontend`（`node:24-<固定版本>-slim`，`npm ci && npm run build`）→ `base`（`python:3.12-slim`，從 `ghcr.io/astral-sh/uv:0.13.1` 複製 uv，`uv sync --frozen --no-dev --no-install-project`）→ `web`（複製 app 與 frontend/dist，`CMD gunicorn -c docker/gunicorn.conf.py "app.main:create_app()"`）→ `worker`（`uv sync --frozen --no-dev --group ocr`，`CMD python -m app.kb.worker_main`）；LiteParse 需要的系統套件在 build 時查明並安裝；非 root 使用者執行
  - `docker/compose.yml`：規格 §3.8 的服務表；named volumes `phoebe-data`、`chroma-data`、`hf-cache`；`web` 依賴 `chroma`；`worker` 與 `vllm` 要求 NVIDIA GPU；vllm 的 8000 對應主機 8001；chroma 映像版本與 `uv.lock` 中 chromadb 的版本一致
  - `.dockerignore`：`.venv`、`node_modules`、`frontend/dist`、`data`、`.git`、`.superpowers`、`__pycache__`、`.env`、`dev_history`
  - `docker/README.md`：前置需求（Docker Desktop、WSL2、NVIDIA 驅動）、`.env` 準備、啟動／停止／看 log、vLLM profile 與 6GB 顯存取捨、主機 Ollama 連線與 `OLLAMA_HOST=0.0.0.0` 排錯、資料備份（volume）
  - 根目錄 `README.md`：新增「設定（.env）」與「部署」小節（連到 `docker/README.md`），快速測試流程改為先 `cp .env.example .env`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gunicorn_conf.py
def test_gunicorn_conf_values(monkeypatch):   # runpy.run_path("docker/gunicorn.conf.py") → bind/worker_class/timeout/graceful_timeout/keepalive；WEB_CONCURRENCY=3 → workers == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gunicorn_conf.py -q`
Expected: FAIL

- [ ] **Step 3: Implement** Dockerfile、compose、設定檔、文件

- [ ] **Step 4: Verify locally**

Run: `uv run pytest -q`；`cd frontend && npm test && npm run lint && npm run typecheck && npm run build`
Expected: 全部通過

- [ ] **Step 5: Verify Docker**

```bash
docker compose -f docker/compose.yml config -q
docker compose -f docker/compose.yml --profile vllm config -q
docker compose -f docker/compose.yml up -d --build
docker compose -f docker/compose.yml ps
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/        # 200
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/me  # 401
```
再用 httpx 腳本：註冊、登入、建立對話、送一則訊息（主機 Ollama）收到 done；建立知識庫、上傳 `tests/pdf_factory.py` 產生的小 PDF、指定 LiteParse，輪詢到 `ready`。確認 `docker compose logs worker` 有處理紀錄、`web` 有 gunicorn worker 啟動紀錄。結束後 `docker compose -f docker/compose.yml down`（保留 volume）。vLLM profile 只驗證 `config`，不實際啟動（6GB 顯存與主機 Ollama 衝突）。

- [ ] **Step 6: Commit**

```bash
git add docker .dockerignore pyproject.toml uv.lock README.md tests/test_gunicorn_conf.py
git commit -m "feat: Docker deployment with Gunicorn web and standalone ingest worker"
```

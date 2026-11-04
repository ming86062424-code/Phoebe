# 本機開發與測試

一般使用請用 Docker 部署（見 [README](../README.md#快速開始docker)）。這份文件給要修改程式的人：在本機直接執行 web 與 ingest、跑測試。

## 事前準備

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Node 22 以上（前端建置與開發用）
- 聊天模型（擇一）：Gemini API key（`GEMINI_API_KEY`），或本機 Ollama（http://localhost:11434，`ollama pull gemma4:e4b`）
- Docker：Milvus、embedder 這些服務在本機開發時也用 Docker 執行

## 本機開發模式（搭配 Docker 的模型服務）

改程式時不必每次重建 Docker 映像：web 與 ingest 直接在本機執行（改了程式會自動重新載入），Milvus、embedder、vLLM 這些重的服務繼續用 Docker 裡的。

**1. 啟動 Docker 裡的服務，並停掉 Docker 版的 web 與 ingest**（它們佔用 8000 埠，也會跟本機版搶工作）：

```
cd docker
docker compose up -d milvus embedder                # 必要：向量資料庫與 embedding
docker compose --profile lightonocr up -d vllm      # 選用：要測 LightOnOCR 時
docker compose stop web ingest
```

**2. 根目錄的 `.env`**（`cp .env.example .env`；本機版讀這個檔，與 `docker/.env` 無關）至少設定：

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

**3. 安裝套件，兩個終端機各跑一個指令：**

```
uv sync
cd frontend && npm install && cd ..

uv run uvicorn app.main:create_app --factory --reload
cd frontend && npm run dev
```

開啟 http://localhost:5173。改後端程式，uvicorn 會自動重啟；改前端程式，畫面會立即更新。前端的 `/api` 請求會轉送到 8000 埠的後端。

**注意**

- 本機版的資料在專案根目錄的 `data/`（SQLite `data/app.db` 與上傳檔 `data/kb/`，可用 `DB_PATH`、`KB_DIR` 覆寫），和 Docker 版的資料完全分開，帳號也要另外建立（管理員帳號一樣由 `.env` 自動建立）。
- LiteParse 在本機（Windows）也能直接使用，`uv sync` 會一併安裝。
- 開發完要回到 Docker 版時：`cd docker && docker compose up -d`（要用 LightOnOCR 則加 `--profile lightonocr`）。
- Gunicorn 只在 Linux 容器內使用；本機一律用 uvicorn。

### 不用前端開發伺服器

先建置前端，再只開 uvicorn，開啟 http://localhost:8000：

```
cd frontend && npm run build
uv run uvicorn app.main:create_app --factory
```

後端在啟動時才決定要不要提供 `frontend/dist`，所以第一次執行 `npm run build` 之後，必須重新啟動 uvicorn。

### 在本機用 transformers 跑 LightOnOCR

沒有 vLLM 時，可以改用 transformers 直接在本機執行 LightOnOCR（Docker 的 ingest 不支援這種方式）。需要額外的 `ocr` 套件群組（torch CUDA 12.6、transformers，約 3GB），並以該群組啟動伺服器：

```
uv sync --group ocr
uv run --group ocr uvicorn app.main:create_app --factory
```

`.env` 設 `OCR_BACKEND=transformers`。未以 `--group ocr` 啟動時，介面會把 LightOnOCR 顯示為「不可用」。第一次使用會從 Hugging Face 下載模型（約 1.8GB）。OCR 開始前會請 Ollama 卸載常駐模型以騰出 VRAM，處理佇列清空後會釋放 OCR 模型。

手動驗證 OCR：

```
uv run --group ocr python -m scripts.ocr_smoke 路徑/檔案.pdf
```

### 重新處理既有文件（本機）

```
uv run python -m app.kb.reprocess_all --dry-run   # 只列出每個知識庫會排入幾份文件
uv run python -m app.kb.reprocess_all
```

指令會把所有「完成」與「失敗」的文件設為排隊中（清除錯誤與 chunk 數），由 ingest 依原解析器重新處理；正在處理中的文件會略過，待其完成後再執行一次即可。Docker 的步驟見 [docker/README.md](../docker/README.md)。

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

用固定的情境實際跑文件處理、搜尋、對話，報告寫到 `docs/benchmarks/`，並和基準比較，說明見 [benchmarks.md](benchmarks.md)。

**對話壓力測試**（僅手動執行；1、2、5、10 人同時提問，最多 10 人，會用到聊天模型的額度）：

```
uv run python -m scripts.loadtest
```

報告寫到 `data/loadtest-<時間>.md`，說明見 [loadtest.md](loadtest.md)。

注意：單純執行 `uv sync` 會移除 `ocr` 群組的套件；`uv run --group ocr` 會重新安裝。

## 開發流程

- 功能、修正、文件分別在 `feat/*`、`fix/*`、`docs/*` 分支開發（從 `develop` 開出），完成後合併回 `develop`。
- 要發布時從 `develop` 開 `release/YYYY-MM-DD` 分支，對 `main` 開 Pull Request，由 repo 擁有者審查後合併。
- 不直接推送到 `main`。

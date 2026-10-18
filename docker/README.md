# Docker 部署

服務：`web`（Gunicorn）、`worker`（獨立文件解析，含 GPU OCR）、`chroma`（向量資料庫）；選用 `vllm`（profile）。所有資料都放在 named volume：`phoebe-data`（SQLite、上傳檔案）、`chroma-data`、`hf-cache`（Hugging Face 模型快取）。

## 前置需求

- Windows：Docker Desktop（使用 WSL2 後端）
- NVIDIA 驅動（支援 WSL2 的版本）；用 `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi` 確認容器看得到 GPU
- 主機已啟動 Ollama，並已下載對話與 Embedding 模型
- Compose 2.24 以上（`env_file` 的 `required: false`）

## 準備 .env

```
cp .env.example .env
```

必須先建立 `.env`（可直接複製範本不修改）：所有指令都帶 `--env-file .env`，因為 Compose 的 `${...}` 變數（`PHOEBE_PORT`、`OCR_MODEL`、`VLLM_GPU_UTIL`）只會從 `--env-file` 讀取，不會自動讀根目錄 `.env`（自動讀取的是 `docker/` 目錄）。容器內會自動覆寫路徑、`CHROMA_HOST=chroma`、`OLLAMA_BASE_URL=http://host.docker.internal:11434` 與 `KB_WORKER_MODE=external`，不需在 `.env` 另外設定。常用項目：`PHOEBE_PORT`（對外連接埠）、`WEB_CONCURRENCY`（Gunicorn worker 數）、`CHAT_MODEL`、`OCR_BACKEND`。

## 啟動、停止、看 log

所有指令都在專案根目錄執行，並帶 `--env-file .env`：

```
docker compose --env-file .env -f docker/compose.yml up -d --build   # 啟動（首次建置含 torch，需要數分鐘與數 GB 磁碟）
docker compose --env-file .env -f docker/compose.yml ps
docker compose --env-file .env -f docker/compose.yml logs -f web worker
docker compose --env-file .env -f docker/compose.yml down            # 停止並移除容器，保留 volume
```

啟動後開啟 http://localhost:8000。`down -v` 會連資料一起刪除，請勿隨意使用。

## vLLM 模式（選用）

預設 `OCR_BACKEND=transformers`：worker 直接在容器內用 transformers 跑 LightOnOCR，OCR 前會請 Ollama 卸載模型以騰出 VRAM。

改用 vLLM：在 `.env` 設定

```
OCR_BACKEND=vllm
VLLM_BASE_URL=http://vllm:8000/v1
```

再以 profile 啟動：

```
docker compose --env-file .env -f docker/compose.yml --profile vllm up -d
```

vLLM 對主機開放在 http://localhost:8001（容器內為 8000）。

6GB 顯存（RTX 2060）的取捨：

- vLLM 會常駐並佔用 `VLLM_GPU_UTIL`（預設 0.35，約 2GB）的顯存，與主機 Ollama 的聊天模型搶 VRAM；聊天可能變慢，或需改用較小的模型。調低 `VLLM_GPU_UTIL` 可留更多顯存給 Ollama，但太低 vLLM 會無法啟動。
- RTX 2060 是 Turing 架構，不支援 bfloat16，因此固定使用 `--dtype float16`。
- 映像固定為 `vllm/vllm-openai:v0.11.2`。該 vLLM 版本是否支援 LightOnOCR 模型、能否在此硬體上運作，尚未驗證；若失敗請改用較新的版本（同時修改 `compose.yml`）或回到 transformers 模式。

## 主機 Ollama 連線

容器以 `http://host.docker.internal:11434` 連到主機 Ollama。若連不上（聊天回錯誤、Embedding 失敗），多半是 Ollama 只監聽 `127.0.0.1`：

1. 設定系統環境變數 `OLLAMA_HOST=0.0.0.0`（或 `0.0.0.0:11434`）
2. 完全結束並重新啟動 Ollama
3. 允許 Windows 防火牆對 Ollama 的私人網路連線
4. 驗證：`docker compose --env-file .env -f docker/compose.yml exec web python -c "import urllib.request;print(urllib.request.urlopen('http://host.docker.internal:11434').read())"`

## 資料備份

資料在 volume 中。備份 `phoebe-data`（建議先 `down`，確保 SQLite 與 Chroma 一致）：

```
docker run --rm -v phoebe_phoebe-data:/data -v "$PWD":/backup alpine tar czf /backup/phoebe-data.tgz -C /data .
docker run --rm -v phoebe_chroma-data:/data -v "$PWD":/backup alpine tar czf /backup/chroma-data.tgz -C /data .
```

還原時把 `tar xzf` 解回空的 volume。volume 的實際名稱可用 `docker volume ls` 確認（前綴為專案名稱 `phoebe_`）。`hf-cache` 只是模型快取，可不備份。

## 注意

- Gunicorn 只在 Linux 容器內使用；Windows 本機快速測試請用 `uv run uvicorn app.main:create_app --factory`。
- 映像版本皆已固定；升級 Chroma 時請與 `uv.lock` 中的 chromadb 版本一致（目前 1.5.9）。
- LiteParse 為 Rust 原生套件，內含 pdfium 與 Tesseract，不需另外安裝 Node 或系統 Tesseract。

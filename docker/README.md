# Docker 部署

服務：`web`（Gunicorn）、`worker`（獨立文件解析，含 GPU OCR）、`chroma`（向量資料庫）；選用 `vllm`（profile）。所有資料都放在 named volume：`phoebe-data`（SQLite、上傳檔案）、`chroma-data`、`hf-cache`（Hugging Face 模型快取）。

## 前置需求

- Windows：Docker Desktop（使用 WSL2 後端）
- NVIDIA 驅動（支援 WSL2 的版本）；用 `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi` 確認容器看得到 GPU
- 主機已啟動 Ollama，並已下載對話與 Embedding 模型
- Compose 2.24 以上（`env_file` 的 `required: false`）

## 準備 docker/.env

Docker 使用自己的設定檔 `docker/.env`（與根目錄的 `.env` 無關，後者只給本機快速測試）。所有指令都在 `docker/` 目錄執行：

```
cd docker
cp .env.example .env
```

編輯 `docker/.env`（可不修改直接使用）。Compose 會自動讀取它做變數插值（`PHOEBE_PORT`、`OCR_MODEL`、`VLLM_GPU_UTIL`），同時也當作容器的環境變數。`DB_PATH`、`KB_DIR`、`CHROMA_HOST/PORT`、`OLLAMA_BASE_URL`、`KB_WORKER_MODE`、`VLLM_BASE_URL`、`HF_HOME` 由 `docker-compose.yml` 固定，無法在 `.env` 覆寫。`docker/.env` 已被 git 忽略，也不會進入映像。

## 啟動、停止、看 log

```
docker compose up -d --build   # 啟動（首次建置含 torch，需要數分鐘與數 GB 磁碟）
docker compose ps
docker compose logs -f web worker
docker compose down            # 停止並移除容器，保留 volume
```

啟動後開啟 http://localhost:8000。`down -v` 會連資料一起刪除，請勿隨意使用。

## vLLM 模式（選用）

預設 `OCR_BACKEND=transformers`：worker 直接在容器內用 transformers 跑 LightOnOCR，OCR 前會請 Ollama 卸載模型以騰出 VRAM。

改用 vLLM：在 `docker/.env` 設定 `OCR_BACKEND=vllm`（worker 連線位址 `http://vllm:8000/v1` 已由 compose 固定）。

再以 profile 啟動：

```
docker compose --profile vllm up -d
```

vLLM 對主機開放在 http://localhost:8001（容器內為 8000）。

本機快速測試也可只用 Docker 跑 vLLM：`cd docker && docker compose --profile vllm up -d vllm`，並在根目錄 `.env` 設 `OCR_BACKEND=vllm`、`VLLM_BASE_URL=http://localhost:8001/v1`。

6GB 顯存（RTX 2060）的取捨：

- vLLM 會常駐並佔用 `VLLM_GPU_UTIL`（預設 0.35，約 2GB）的顯存，與主機 Ollama 的聊天模型搶 VRAM；聊天可能變慢，或需改用較小的模型。調低 `VLLM_GPU_UTIL` 可留更多顯存給 Ollama，但太低 vLLM 會無法啟動。
- RTX 2060 是 Turing 架構，不支援 bfloat16，因此固定使用 `--dtype float16`。
- 映像固定為 `vllm/vllm-openai:v0.30.0`（v0.11.2 尚不支援 Qwen3.5 架構）。啟動參數依 LightOnOCR 官方模型卡：`--trust-remote-code` 與 `--default-chat-template-kwargs '{"enable_thinking": false}'`（模型以關閉 thinking 訓練與評估）。能否在此硬體上運作尚未驗證；若失敗請回到 transformers 模式。
- 依官方用法，OCR 使用取樣（`temperature=0.2`；vLLM 另加 `top_p=0.9`），因此重跑同一份 PDF 的結果可能有些微差異（這取代了先前的貪婪解碼，依 repo 擁有者要求遵循官方用法）。頁面以 400 DPI 算繪並縮圖至最長邊 2048。

## 主機 Ollama 連線

容器以 `http://host.docker.internal:11434` 連到主機 Ollama。若連不上（聊天回錯誤、Embedding 失敗），多半是 Ollama 只監聽 `127.0.0.1`：

1. 設定系統環境變數 `OLLAMA_HOST=0.0.0.0`（或 `0.0.0.0:11434`）
2. 完全結束並重新啟動 Ollama
3. 允許 Windows 防火牆對 Ollama 的私人網路連線
4. 驗證：`docker compose exec web python -c "import urllib.request;print(urllib.request.urlopen('http://host.docker.internal:11434').read())"`

## 資料備份

資料在 volume 中。備份 `phoebe-data`（建議先 `down`，確保 SQLite 與 Chroma 一致）。以下指令請在 PowerShell 執行；若用 Git Bash，需在每個指令前加 `MSYS_NO_PATHCONV=1`，否則 `-C /data` 與 `/backup` 會被轉成 Windows 路徑：

```
docker run --rm -v phoebe_phoebe-data:/data -v "$PWD":/backup alpine tar czf /backup/phoebe-data.tgz -C /data .
docker run --rm -v phoebe_chroma-data:/data -v "$PWD":/backup alpine tar czf /backup/chroma-data.tgz -C /data .
```

還原時把 `tar xzf` 解回空的 volume。volume 的實際名稱可用 `docker volume ls` 確認（前綴為專案名稱 `phoebe_`）。`hf-cache` 只是模型快取，可不備份。

## 注意

- Gunicorn 只在 Linux 容器內使用；Windows 本機快速測試請用 `uv run uvicorn app.main:create_app --factory`。
- 映像版本皆已固定；升級 Chroma 時請與 `uv.lock` 中的 chromadb 版本一致（目前 1.5.9）。
- LiteParse 為 Rust 原生套件，內含 pdfium 與 Tesseract，不需另外安裝 Node 或系統 Tesseract。預設語言（`chi_tra`、`chi_tra_vert`、`eng`，來自 tessdata_best 的固定 commit）已內建於 worker 映像；`LITEPARSE_OCR_LANGUAGE` 改用其他語言時，該語言檔會在第一次 OCR 時下載（需要對外網路，且存在容器可寫層，重建映像後需重新下載）。
- 若先以 `--profile vllm` 啟動過，`hf-cache` volume 可能由 vLLM 容器以 root 建立，導致 worker（uid 1000）無法寫入模型快取（下載模型時出現 Permission denied）。修正：`docker run --rm -v phoebe_hf-cache:/hf-cache alpine chown -R 1000:1000 /hf-cache`（volume 名稱以 `docker volume ls` 為準）。
- worker 映像內含 `gcc` 與 `libc6-dev`：Linux 版 torch 附帶 Triton，Qwen3.5（LightOnOCR）執行時會即時編譯 Triton kernel，沒有 C 編譯器會出現「Failed to find C compiler」而 OCR 失敗。編譯快取放在 `/home/phoebe/.triton`（`TRITON_CACHE_DIR`），家目錄歸 `phoebe` 所有。web 映像不需要，因此不含編譯器。
- 映像大小：worker 預期約 7–8 GB（CUDA 函式庫約 3.6 GB、torch 約 1.7 GB、triton 約 0.9 GB），web 映像小得多。uv 的下載快取放在 BuildKit cache mount（`/root/.cache/uv`），不會進入映像層；`docker builder prune` 可清除它。程式碼以 `COPY --chown` 放入，不做遞迴 `chown`，避免複製整棵目錄樹。
- 根目錄 `.dockerignore` 是白名單（先 `*` 全部排除，再用 `!` 放行 build 需要的檔案）。Dockerfile 新增 `COPY` 來源時，必須同步在 `.dockerignore` 明確放行（`tests/test_config.py` 會檢查）。

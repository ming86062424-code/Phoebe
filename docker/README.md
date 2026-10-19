# Docker 部署

服務：`web`（Gunicorn）、`worker`（獨立文件解析，不含 torch）、`vllm`（LightOnOCR 官方服務方式，GPU 專用於 OCR）、`milvus`（向量資料庫，含 `etcd`、`minio`；Milvus 映像 `v2.6.25`）。所有資料都放在 named volume：`phoebe-data`（SQLite、上傳檔案）、`milvus-data`、`etcd-data`、`minio-data`、`hf-cache`（vLLM 的 Hugging Face 模型快取）；舊的 `chroma-data` 僅供遷移。

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

編輯 `docker/.env`（可不修改直接使用）。Compose 會自動讀取它做變數插值（`PHOEBE_PORT`、`OCR_MODEL`、`OCR_VLLM_GPU_UTIL`、`OCR_VLLM_MAX_MODEL_LEN`、`OCR_VLLM_MAX_NUM_SEQS`），同時也當作容器的環境變數（vllm 除外，它不載入 `docker/.env`）。`DB_PATH`、`KB_DIR`、`MILVUS_URI`、`OLLAMA_BASE_URL`、`KB_WORKER_MODE`、`OCR_BACKEND`、`VLLM_BASE_URL` 由 `docker-compose.yml` 固定，無法在 `.env` 覆寫。`docker/.env` 已被 git 忽略，也不會進入映像。

## 啟動、停止、看 log

```
docker compose up -d --build   # 啟動（首次啟動 vllm 要下載映像與模型，約需數 GB 磁碟與數分鐘；worker 在 vllm 容器啟動後就會啟動；vLLM 載入完成前 LightOnOCR 顯示為無法使用，LiteParse 照常運作）
docker compose ps
docker compose logs -f web worker
docker compose down            # 停止並移除容器，保留 volume
```

啟動後開啟 http://localhost:8000。`down -v` 會連資料一起刪除，請勿隨意使用。

## OCR：vLLM（預設）

LightOnOCR 依官方方式用 vLLM 服務（映像 `vllm/vllm-openai:v0.30.0`，v0.11.2 尚不支援 Qwen3.5 架構）。GPU 專用於 OCR 模型：對話模型可以放在雲端或主機 Ollama，因此 `OCR_VLLM_GPU_UTIL` 預設 0.8，不再替 Ollama 保留顯存。worker 固定使用 `OCR_BACKEND=vllm`、`VLLM_BASE_URL=http://vllm:8000/v1`，不含 torch，也不需要 GPU。

啟動參數（0.8B 官方設定）：`--trust-remote-code`、`--limit-mm-per-prompt '{"image": 1}'`、`--default-chat-template-kwargs '{"enable_thinking": false}'`、`--mm-processor-cache-gb 0`、`--no-enable-prefix-caching`、`--max-model-len 18000`（`OCR_VLLM_MAX_MODEL_LEN`）。RTX 2060 是 Turing 架構，不支援 bfloat16，因此固定 `--dtype float16`。

**調校（RTX 2060 6GB 實測）**

| 設定 | 預設 | 說明 |
|---|---|---|
| `OCR_VLLM_GPU_UTIL` | `0.8` | 0.85 失敗：可用顯存 4.98 GiB < 需要的 5.1 GiB（Windows 桌面約佔 1 GiB）；0.8 可以啟動 |
| `--enforce-eager`（固定） | 開 | CUDA graph 的額外記憶體會讓 KV cache 配不到空間（「No available memory for the cache blocks」），改用 eager 模式 |
| `OCR_VLLM_MAX_NUM_SEQS` | `8` | vLLM 預設 256 條序列 × 18000 tokens 在 6GB 配不到 KV cache；8 條時 KV cache 1.28 GiB（99,243 tokens，18000 長度下約 5.5 倍並行），GPU 總用量約 4.4/6 GiB |

顯存更大的 GPU 可調高 `OCR_VLLM_GPU_UTIL` 與 `OCR_VLLM_MAX_NUM_SEQS`，並視需要移除 compose 中的 `--enforce-eager` 以啟用 CUDA graph（吞吐量較高）。Compose 變數刻意不用 `VLLM_` 開頭，否則 vLLM 會警告未知的環境變數（vllm 服務也不載入 `docker/.env`）。

**驅動需求**：`vllm/vllm-openai:v0.30.0` 是 CUDA 13 映像，NVIDIA 驅動需 580 以上（實測 580.88）。

**升級提醒**：舊的 `docker/.env` 若有 `VLLM_GPU_UTIL`、`VLLM_MAX_MODEL_LEN`，請改名為 `OCR_VLLM_GPU_UTIL`、`OCR_VLLM_MAX_MODEL_LEN`（舊名稱已不會被讀取），也可參考 `.env.example` 新增 `OCR_VLLM_MAX_NUM_SEQS`。

- 模型快取：`hf-cache` volume 掛在 `/root/.cache/huggingface`（vLLM 映像以 root 執行、未設 `HF_HOME` 時的預設位置）。
- vllm 有 healthcheck（`GET /health`，`start_period` 600 秒涵蓋首次下載與載入模型）；worker 只等它啟動（`service_started`）、不等健康：vLLM 還在下載模型或啟動失敗時，LiteParse 仍可使用，LightOnOCR 則顯示「OCR 服務（vLLM）未啟動」。
- 頁面以 400 DPI 算繪並縮圖至最長邊 2048。請求只含頁面影像，`temperature=0.1`、`chat_template_kwargs={"enable_thinking": false}`、`max_tokens=4096`（官方建議不要用貪婪解碼，所以同一份 PDF 重跑結果可能有些微差異）。OCR 結果中被跳脫的數學分隔符號（反斜線加錢字號）會還原成一般的 `$`，單獨出現的金額（如 `NT` 加跳脫錢字號）不變。
- 本機快速測試（Windows 不能直接跑 vLLM）可只啟動 vllm：`cd docker && docker compose up -d vllm`，並在根目錄 `.env` 設 `OCR_BACKEND=vllm`、`VLLM_BASE_URL=http://localhost:8001/v1`（vllm 只對本機 `127.0.0.1:8001` 開放，因為它沒有驗證）。
- 沒有 GPU 或想用 transformers 時，只能在本機快速測試用 `uv sync --group ocr` 搭配 `OCR_BACKEND=transformers`；Docker 的 worker 不含 torch。

## Milvus 與舊資料遷移

`milvus` 只對本機 `127.0.0.1:19530` 開放（無驗證）；`web`、`worker` 等它健康後才啟動（首次啟動約需 90 秒）。`MILVUS_TOKEN`、`MILVUS_COLLECTION` 可在 `docker/.env` 設定，`MILVUS_URI` 固定為 `http://milvus:19530`。

不啟動應用程式容器的快速測試：`cd docker && docker compose up -d milvus`（會一併啟動 etcd、minio），並在根目錄 `.env` 設 `MILVUS_URI=http://localhost:19530`。

從舊版 Chroma 遷移資料（一次性；`chroma-data` volume 需仍存在）：

```
cd docker
docker compose up -d milvus
docker compose --profile migrate-chroma up -d chroma
docker compose run --rm --no-deps worker python -m app.kb.migrate_chroma --chroma-host chroma
docker compose --profile migrate-chroma stop chroma
```

（`--no-deps` 且先單獨啟動 `milvus`，避免連帶啟動 vllm、web。）

本機（非 Docker）遷移：

```
uv run --group migrate python -m app.kb.migrate_chroma --chroma-dir data/chroma
```

若之後刪除或重建 Milvus collection（例如更換 `EMBED_MODEL` 或斷詞設定），請重新啟動 web 與 worker：它們會在程序存活期間快取「collection 已就緒」的狀態。

可加 `--dry-run` 先檢查。worker 映像已包含 `migrate` 依賴群組（chromadb）。遷移完成後 `chroma-data` volume 可自行刪除。

## 管理員與 Gemini 聊天

在 `docker/.env` 設定（修改後需 `docker compose up -d` 重新建立容器才會套用）：

- `ADMIN_USERNAMES=alice,bob`：管理員帳號，逗號分隔，大小寫需完全相符。管理員可看到「系統設定」頁，設定LLM 供應商（Ollama／Gemini）、模型（即時取得清單）、系統提示詞（最多 4000 字）與 temperature（自訂或預設值）。
- `GEMINI_API_KEY`：到 [Google AI Studio](https://aistudio.google.com/apikey) 建立金鑰後填入。金鑰只放在 `docker/.env`，不會出現在介面或資料庫；留空則 Gemini 選項停用。
- `GEMINI_MODEL`：預設 `gemini-2.5-flash`。

GPU 專用於 OCR（vLLM）時，建議在「系統設定」把聊天切換為 Gemini。Embedding 仍使用主機 Ollama。

## 主機 Ollama 連線

容器以 `http://host.docker.internal:11434` 連到主機 Ollama。若連不上（聊天回錯誤、Embedding 失敗），多半是 Ollama 只監聽 `127.0.0.1`：

1. 設定系統環境變數 `OLLAMA_HOST=0.0.0.0`（或 `0.0.0.0:11434`）
2. 完全結束並重新啟動 Ollama
3. 允許 Windows 防火牆對 Ollama 的私人網路連線
4. 驗證：`docker compose exec web python -c "import urllib.request;print(urllib.request.urlopen('http://host.docker.internal:11434').read())"`

## 資料備份

資料在 volume 中。備份 `phoebe-data`（建議先 `down`，確保 SQLite 與 Milvus 一致）。以下指令請在 PowerShell 執行；若用 Git Bash，需在每個指令前加 `MSYS_NO_PATHCONV=1`，否則 `-C /data` 與 `/backup` 會被轉成 Windows 路徑：

```
docker run --rm -v phoebe_phoebe-data:/data -v "$PWD":/backup alpine tar czf /backup/phoebe-data.tgz -C /data .
docker run --rm -v phoebe_milvus-data:/data -v "$PWD":/backup alpine tar czf /backup/milvus-data.tgz -C /data .
docker run --rm -v phoebe_etcd-data:/data -v "$PWD":/backup alpine tar czf /backup/etcd-data.tgz -C /data .
docker run --rm -v phoebe_minio-data:/data -v "$PWD":/backup alpine tar czf /backup/minio-data.tgz -C /data .
```

還原時把 `tar xzf` 解回空的 volume。volume 的實際名稱可用 `docker volume ls` 確認（前綴為專案名稱 `phoebe_`）。`hf-cache` 只是模型快取，可不備份。

## 注意

- Gunicorn 只在 Linux 容器內使用；Windows 本機快速測試請用 `uv run uvicorn app.main:create_app --factory`。
- 映像版本皆已固定；升級 Milvus 時請與 `uv.lock` 中的 pymilvus 版本相容（目前 pymilvus 2.6.17 對應 Milvus v2.6.25）；etcd、minio 沿用該版官方 standalone compose。`chroma` 映像（1.5.9）只供遷移，須與 chromadb 版本一致。
- LiteParse 為 Rust 原生套件，內含 pdfium 與 Tesseract，不需另外安裝 Node 或系統 Tesseract。預設語言（`chi_tra`、`chi_tra_vert`、`eng`，來自 tessdata_best 的固定 commit）已內建於 worker 映像；`LITEPARSE_OCR_LANGUAGE` 改用其他語言時，該語言檔會在第一次 OCR 時下載（需要對外網路，且存在容器可寫層，重建映像後需重新下載）。
- 映像大小：worker 與 web 都不含 torch，各約 1 GB（worker 另含遷移用的 chromadb）；vllm 是官方映像（數 GB）。uv 的下載快取放在 BuildKit cache mount（`/root/.cache/uv`），不會進入映像層；`docker builder prune` 可清除它。程式碼以 `COPY --chown` 放入，不做遞迴 `chown`，避免複製整棵目錄樹。
- 根目錄 `.dockerignore` 是白名單（先 `*` 全部排除，再用 `!` 放行 build 需要的檔案）。Dockerfile 新增 `COPY` 來源時，必須同步在 `.dockerignore` 明確放行（`tests/test_config.py` 會檢查）。

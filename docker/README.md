# Docker 部署

服務：`web`（Gunicorn）、`ingest`（獨立文件解析，不含 torch）、`vllm`（LightOnOCR 官方服務方式，GPU 專用於 OCR）、`milvus`（向量資料庫，含 `etcd`、`minio`；Milvus 映像 `v2.6.25`）、`embedder`（embedding 服務，見下）。每個服務的功能、互相呼叫的方式與資料流程見 [docs/services.md](../docs/services.md)。資料位置：

| 位置 | 內容 |
|---|---|
| 主機目錄 `docker/volumes/kb/` | 上傳的檔案與解析出的圖片 |
| volume `phoebe_phoebe-db` | SQLite 資料庫 `app.db` |
| volume `phoebe_milvus-data`、`phoebe_etcd-data`、`phoebe_minio-data` | Milvus 向量資料庫 |
| volume `phoebe_embedder-hf-cache` | embedder 的模型快取 |
| volume `phoebe_vllm-hf-cache` | vLLM（LightOnOCR）的 Hugging Face 模型快取 |
| volume `phoebe_vllm-triton-cache` | vLLM 的 Triton 編譯快取 |

資料庫與快取放在 Docker 管理的 volume，使用者會想直接查看的上傳檔案放在主機目錄 `docker/volumes/`（第一次啟動時自動建立，已被 git 忽略）。為什麼這樣分、想改回主機目錄怎麼做，見 [docs/docker-storage.md](../docs/docker-storage.md)。

**從舊版升級（2026-10-24 以前，Milvus 與模型快取放在 `docker/volumes/` 下）**：新版改讀 Docker 管理的 volume，不搬移資料的話知識庫向量會是空的、模型會重新下載。搬移步驟見 [docs/docker-storage.md](../docs/docker-storage.md#從舊版主機目錄搬到-docker-管理的-volume)。

## 前置需求

- Windows：Docker Desktop（使用 WSL2 後端）
- NVIDIA 驅動（支援 WSL2 的版本）；用 `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi` 確認容器看得到 GPU
- 主機已啟動 Ollama，並已下載對話模型（Embedding 由 `embedder` 容器提供，不需 Ollama）
- Compose 2.24 以上（`env_file` 的 `required: false`）

## 準備 docker/.env

Docker 使用自己的設定檔 `docker/.env`（與根目錄的 `.env` 無關，後者只給本機快速測試）。所有指令都在 `docker/` 目錄執行：

```
cd docker
cp .env.example .env
```

編輯 `docker/.env`（可不修改直接使用）。Compose 會自動讀取它做變數插值（`PHOEBE_PORT`、`OCR_MODEL`、`OCR_VLLM_KV_CACHE_MEMORY`、`OCR_VLLM_GPU_UTIL`、`OCR_VLLM_MAX_MODEL_LEN`、`OCR_VLLM_MAX_NUM_SEQS`），同時也當作容器的環境變數（vllm 除外，它不載入 `docker/.env`）。`DB_PATH`、`KB_DIR`、`MILVUS_URI`、`OLLAMA_BASE_URL`、`KB_INGEST_MODE`、`OCR_BACKEND`、`VLLM_BASE_URL` 由 `docker-compose.yml` 固定，無法在 `.env` 覆寫。`docker/.env` 已被 git 忽略，也不會進入映像。

## 啟動、停止、看 log

```
docker compose up -d --build   # 啟動（不含 vLLM；LightOnOCR 顯示為無法使用，LiteParse 照常運作）
docker compose --profile lightonocr up -d   # 連 vLLM 一起啟動（首次要下載映像與模型，約需數 GB 磁碟與數分鐘；載入完成前 LightOnOCR 仍顯示無法使用）
docker compose ps
docker compose logs -f web ingest
docker compose down            # 停止並移除容器，資料（docker/volumes 與各個 volume）都會保留
docker compose --profile lightonocr down   # 連 vLLM 一起停止
```

注意：不加 `--profile lightonocr` 的 `docker compose down` 不會停止 vLLM，已啟動的 vllm 容器會繼續佔用 GPU。

啟動後開啟 http://localhost:8000。`down -v` 會連資料一起刪除，請勿隨意使用。

## OCR：vLLM（選用，`lightonocr` profile）

vllm 服務預設不啟動，需用 `docker compose --profile lightonocr up -d`。ingest 對 vllm 的依賴是 `required: false`，沒啟動 vLLM 時 ingest 照常啟動。

LightOnOCR 依官方方式用 vLLM 服務（映像 `vllm/vllm-openai:v0.30.0`，v0.11.2 尚不支援 Qwen3.5 架構）。顯存用量由 KV cache 的大小決定（`OCR_VLLM_KV_CACHE_MEMORY`，預設 512M），RTX 2060 上 vLLM 約佔 2.8 GB，設計原因見 [docs/gpu-memory.md](../docs/gpu-memory.md)。ingest 固定使用 `OCR_BACKEND=vllm`、`VLLM_BASE_URL=http://vllm:8000/v1`，不含 torch，也不需要 GPU。

啟動參數（0.8B 官方設定）：`--trust-remote-code`、`--limit-mm-per-prompt '{"image": 1}'`、`--default-chat-template-kwargs '{"enable_thinking": false}'`、`--mm-processor-cache-gb 0`、`--no-enable-prefix-caching`、`--max-model-len 18000`（`OCR_VLLM_MAX_MODEL_LEN`）。RTX 2060 是 Turing 架構，不支援 bfloat16，因此固定 `--dtype float16`。

**調校（RTX 2060 6GB 實測）**：完整的排錯過程、錯誤訊息與和 Ollama 共用 GPU 的說明見 [docs/vllm-rtx2060.md](../docs/vllm-rtx2060.md)。

| 設定 | 預設 | 說明 |
|---|---|---|
| `OCR_VLLM_KV_CACHE_MEMORY` | `512M` | KV cache 約 3.9 萬個 token，足夠同時 OCR 4 頁；2026-10-24 前是用比例 0.8 讓 vLLM 自己算，KV cache 有 1.28 GiB，用不到那麼多 |
| `OCR_VLLM_GPU_UTIL` | `0.7` | 只當作啟動檢查：剩餘顯存少於 0.7 × 6 GB 就拒絕啟動。不指定時預設 0.92，Windows 桌面約佔 1 GiB，6 GB 顯卡會啟動失敗 |
| `--enforce-eager`（固定） | 開 | CUDA graph 的額外記憶體會讓 KV cache 配不到空間（「No available memory for the cache blocks」），改用 eager 模式 |
| `OCR_VLLM_MAX_NUM_SEQS` | `4` | vLLM 預設 256 條序列 × 18000 tokens 在 6GB 配不到 KV cache；改為 4 條後實測：KV cache 1.28 GiB（99,243 tokens，18000 長度下約 5.5 倍並行），GPU 總用量約 4.4/6 GiB，與 8 條時相同 |

顯存更大的 GPU 可調高 `OCR_VLLM_KV_CACHE_MEMORY` 與 `OCR_VLLM_MAX_NUM_SEQS`，並視需要移除 compose 中的 `--enforce-eager` 以啟用 CUDA graph（吞吐量較高）。Compose 變數刻意不用 `VLLM_` 開頭，否則 vLLM 會警告未知的環境變數（vllm 服務也不載入 `docker/.env`）。

**驅動需求**：`vllm/vllm-openai:v0.30.0` 是 CUDA 13 映像，NVIDIA 驅動需 580 以上（實測 580.88）。

**升級提醒**：舊的 `docker/.env` 若有 `VLLM_GPU_UTIL`、`VLLM_MAX_MODEL_LEN`，請改名為 `OCR_VLLM_GPU_UTIL`、`OCR_VLLM_MAX_MODEL_LEN`（舊名稱已不會被讀取），也可參考 `.env.example` 新增 `OCR_VLLM_MAX_NUM_SEQS`。

- 模型快取：volume `phoebe_vllm-hf-cache` 掛在 `/root/.cache/huggingface`（vLLM 映像以 root 執行、未設 `HF_HOME` 時的預設位置）。
- Triton 編譯快取：volume `phoebe_vllm-triton-cache` 掛在 `/root/.triton/cache`。沒保存的話，每次重建容器都要重新編譯好幾分鐘，見 [docs/docker-storage.md](../docs/docker-storage.md#triton-編譯快取)。
- vllm 有 healthcheck（`GET /health`，`start_period` 600 秒涵蓋首次下載與載入模型）；ingest 只等它啟動（`service_started`）、不等健康：vLLM 還在下載模型或啟動失敗時，LiteParse 仍可使用，LightOnOCR 則顯示「OCR 服務（vLLM）未啟動」。
- 頁面以 400 DPI 算繪並縮圖至最長邊 2048。請求只含頁面影像，`temperature=0.1`、`chat_template_kwargs={"enable_thinking": false}`、`max_tokens=4096`（官方建議不要用貪婪解碼，所以同一份 PDF 重跑結果可能有些微差異）。OCR 結果中被跳脫的數學分隔符號（反斜線加錢字號）會還原成一般的 `$`，單獨出現的金額（如 `NT` 加跳脫錢字號）不變。
- 本機快速測試（Windows 不能直接跑 vLLM）可只啟動 vllm：`cd docker && docker compose --profile lightonocr up -d vllm`，並在根目錄 `.env` 設 `OCR_BACKEND=vllm`、`VLLM_BASE_URL=http://localhost:8001/v1`（vllm 只對本機 `127.0.0.1:8001` 開放，因為它沒有驗證）。
- 沒有 GPU 或想用 transformers 時，只能在本機快速測試用 `uv sync --group ocr` 搭配 `OCR_BACKEND=transformers`；Docker 的 ingest 不含 torch。

## Embedding 服務（embedder）

獨立的 FastAPI 服務（`embedder/`，sentence-transformers 6.1.0、transformers 5.19.0（第一個支援 embedding_gemma2 的版本）、torch 2.14.0 與 torchvision 0.29.0 CPU wheel），以 `google/embeddinggemma-2` 同時產生文字與圖片的 768 維向量（L2 normalized；CPU float32，模型卡明言不可用 float16）。只載入文字與視覺部分：模型另有一個 304M 參數的音訊編碼器，我們用不到，載入時以 `audio_config=None` 略過，記憶體少約 1.2 GB，文字與圖片的向量和完整模型完全相同。只對本機 `127.0.0.1:8002` 開放（無驗證）；Docker 內以 `http://embedder:8000` 連線。

- `POST /embed`：`{"items": [{"kind": "query"|"document"|"image"|"image_text", ...}]}` → `{"embeddings": [[...768]]}`；`GET /health`：模型載入完成才回 200，載入中回 503。
- 設定（`docker/.env`）：`EMBED_MODEL_ID`（預設 `google/embeddinggemma-2`）、`EMBED_IMAGE_TOKENS`（每張圖片的 soft token，預設 280）。
- `EMBED_IMAGE_TOKENS` 只接受 70、140、280、560、1120，否則服務啟動失敗。模型載入失敗（例如下載中斷）時程序會直接結束，由 `restart: unless-stopped` 自動重啟重試。服務只傳入 `EMBED_MODEL_ID`、`EMBED_IMAGE_TOKENS`、`HF_HOME`（與主機環境有設定時的 `HF_TOKEN`），不載入整份 `docker/.env`。圖片上限：每張解碼後 20 MB 以內、25,000,000 像素以內，整個請求 100 MB 以內（超過回 422／413）。
- 首次啟動會下載約 1.5 GB 模型到 volume `phoebe_embedder-hf-cache`（healthcheck `start_period` 600 秒）。此 volume 與 vLLM 的模型快取分開，避免 root／非 root 寫入權限衝突。
- 首次下載可能超過 healthcheck 的等待時間：`start_period` 600 秒，之後每 15 秒檢查一次、最多失敗 10 次（約再 150 秒）。`web`、`ingest` 依賴 `embedder` 健康，若在 `docker compose up -d` 期間因此放棄啟動，等 `docker compose ps` 顯示 embedder 為 healthy（`docker compose logs -f embedder` 可看下載進度）後，再執行一次 `docker compose up -d` 即可，已下載的模型快取不會重來。
- 本機快速測試：`cd docker && docker compose up -d embedder`；單元測試：`cd embedder && uv run pytest -q`（用假模型，不下載模型）。

## Milvus 與舊資料遷移

`milvus` 只對本機 `127.0.0.1:19530` 開放（無驗證）；`web`、`ingest` 等它健康後才啟動（首次啟動約需 90 秒）。`MILVUS_TOKEN`、`MILVUS_COLLECTION` 可在 `docker/.env` 設定，`MILVUS_URI` 固定為 `http://milvus:19530`。

不啟動應用程式容器的快速測試：`cd docker && docker compose up -d milvus`（會一併啟動 etcd、minio），並在根目錄 `.env` 設 `MILVUS_URI=http://localhost:19530`。

若之後刪除或重建 Milvus collection（例如更換 embedding 模型或斷詞設定），請重新啟動 web 與 ingest：它們會在程序存活期間快取「collection 已就緒」的狀態。

更早使用 Chroma 的版本：不需要遷移向量，依下方「從舊版升級」重新處理文件即可（原始上傳檔與 SQLite 都還在）。舊的 `chroma-data` volume 已不再使用，可自行刪除：`docker volume rm phoebe_chroma-data`。

## 從舊版升級（worker 改名為 ingest）

文件處理服務從 `worker` 改名為 `ingest`（服務、映像、環境變數都改了），vLLM 也改成選用的 `lightonocr` profile。升級時：

1. 修改 `docker/.env`：
   - `KB_WORKER_POLL_SECONDS` 改為 `KB_INGEST_POLL_SECONDS`（`KB_WORKER_MODE` 若有也一樣改為 `KB_INGEST_MODE`）。**舊名稱仍存在時服務會拒絕啟動**，錯誤訊息會提示新名稱
   - `OCR_VLLM_MAX_NUM_SEQS=8` 改為 `4`，或刪除這行改用新的預設值
   - 視需要加上 `INGEST_CONCURRENCY`、`OCR_CONCURRENCY`、`OCR_TIMEOUT_SECONDS`（說明見 `.env.example`）
2. 第一次啟動新版時加上 `--remove-orphans`，清掉舊的 `worker` 容器；否則它會繼續執行，和新的 ingest 搶同一個佇列：

   ```
   docker compose --profile lightonocr up -d --build --remove-orphans
   ```

3. 之後要用 LightOnOCR，啟動與停止都要帶 `--profile lightonocr`。

## 從舊版升級（多模態知識庫）

多模態版本更換了 embedding 模型（Ollama `embeddinggemma` → `embedder` 服務的 `google/embeddinggemma-2`）並新增 collection 欄位，舊的 `phoebe_chunks` collection 無法沿用（web／ingest 會回報「知識庫結構已更新，請重建知識庫」）。上傳的原始檔與 SQLite 資料都保留，只需重建向量並重新處理文件。在 `docker/` 目錄依序執行：

```
git pull                                   # 取得新版程式與 compose
docker compose up -d --build embedder milvus   # 先建置並啟動 embedder（首次會下載約 1–2 GB 模型）與 milvus
docker compose ps                          # 等 embedder 與 milvus 都是 healthy
docker compose stop web ingest            # 先停掉舊的 web／ingest，避免它們用舊結構重建 collection
# 刪除舊的 collection（名稱為 MILVUS_COLLECTION，預設 phoebe_chunks）
docker compose run --rm --no-deps ingest python -c "from app.config import Settings; from pymilvus import MilvusClient; s = Settings.from_env(); MilvusClient(uri=s.milvus_uri, token=s.milvus_token).drop_collection(s.milvus_collection)"
docker compose up -d --build              # 重建並重新啟動 web、ingest（它們會快取 collection 狀態，必須重啟）
docker compose restart web ingest         # 若容器沒有被重建，也請明確重啟
docker compose run --rm --no-deps ingest python -m app.kb.reprocess_all --dry-run   # 先看每個知識庫會排入幾份文件
docker compose run --rm --no-deps ingest python -m app.kb.reprocess_all             # 把所有文件重新排入佇列
docker compose logs -f ingest             # ingest 依序重新處理
```

說明：

- `run --rm --no-deps ingest` 會開一個一次性容器，掛載同樣的 SQLite（`phoebe-db`）與 `docker/volumes/kb`，連到 Milvus，不會影響執行中的 ingest；若想用執行中的容器，也可改用 `docker compose exec ingest python -m app.kb.reprocess_all`。
- `reprocess_all` 把所有「完成」與「失敗」的文件設為排隊中（清除錯誤與 chunk 數），保留原解析器。正在處理中的文件會略過（避免與 ingest 搶同一份文件），待它們完成後再執行一次即可。舊文件的圖片資料夾由 ingest 在重新解析前自動清除，指令本身不刪檔，以免與剛認領文件的 ingest 競爭。
- 重新處理時 ingest 會逐頁呼叫 embedder（CPU），文件多或頁數多時會花一段時間；PDF 內的圖片在重新處理後才會有縮圖與圖片向量。

## 管理員與 Gemini 聊天

在 `docker/.env` 設定（修改後需 `docker compose up -d` 重新建立容器才會套用）：

- `ADMIN_USERNAME`、`ADMIN_PASSWORD`：管理員帳密，web 啟動時自動建立帳號並同步密碼（改密碼後 `docker compose restart web ingest`）。舊版的 `ADMIN_USERNAMES` 已不再使用。管理員可看到「系統設定」頁，設定LLM 供應商（Ollama／Gemini）、模型（即時取得清單）、系統提示詞（最多 4000 字）與 temperature（自訂或預設值）。
- `GEMINI_API_KEY`：到 [Google AI Studio](https://aistudio.google.com/apikey) 建立金鑰後填入。金鑰只放在 `docker/.env`，不會出現在介面或資料庫；留空則 Gemini 選項停用。
- `GEMINI_MODEL`：預設 `gemini-2.5-flash`。

GPU 專用於 OCR（vLLM）時，建議在「系統設定」把聊天切換為 Gemini。Embedding 一律使用 `embedder` 容器（CPU）。

## 主機 Ollama 連線

容器以 `http://host.docker.internal:11434` 連到主機 Ollama。若連不上（聊天回錯誤），多半是 Ollama 只監聽 `127.0.0.1`：

1. 設定系統環境變數 `OLLAMA_HOST=0.0.0.0`（或 `0.0.0.0:11434`）
2. 完全結束並重新啟動 Ollama
3. 允許 Windows 防火牆對 Ollama 的私人網路連線
4. 驗證：`docker compose exec web python -c "import urllib.request;print(urllib.request.urlopen('http://host.docker.internal:11434').read())"`

## 資料備份

先 `docker compose down`（確保 SQLite 與 Milvus 一致），再備份：

- `docker/volumes/kb/`：直接複製資料夾即可
- SQLite 與 Milvus（`phoebe_phoebe-db`、`phoebe_milvus-data`、`phoebe_etcd-data`、`phoebe_minio-data`）在 Docker 管理的 volume，用下面的指令打包（模型與 Triton 快取可不備份）
- SQLite 在 named volume，以下指令把它打包到目前目錄（PowerShell；Git Bash 需在指令前加 `MSYS_NO_PATHCONV=1`）：

```
docker run --rm -v phoebe_phoebe-db:/db -v "$PWD":/backup alpine tar czf /backup/phoebe-db.tgz -C /db .
docker run --rm -v phoebe_milvus-data:/d -v "$PWD":/backup alpine tar czf /backup/milvus-data.tgz -C /d .
docker run --rm -v phoebe_etcd-data:/d -v "$PWD":/backup alpine tar czf /backup/etcd-data.tgz -C /d .
docker run --rm -v phoebe_minio-data:/d -v "$PWD":/backup alpine tar czf /backup/minio-data.tgz -C /d .
```

還原時把 `kb/` 資料夾放回 `docker/volumes/`，並用 `tar xzf` 把各個備份解回對應的空 volume（例如 `docker run --rm -v phoebe_milvus-data:/d -v "$PWD":/backup alpine tar xzf /backup/milvus-data.tgz -C /d`）。Milvus 的三個 volume 要一起備份、一起還原。

**更早的版本（2026-10-19 以前，全部用 named volume）**：那時的 `phoebe_phoebe-data`、`phoebe_hf-cache` 已不再使用，確定不需要後可刪除：`docker volume rm phoebe_phoebe-data phoebe_hf-cache`。其他同名的 volume（`phoebe_milvus-data` 等）**現在又在使用中，不要刪除**。

## 注意

- Gunicorn 只在 Linux 容器內使用；Windows 本機快速測試請用 `uv run uvicorn app.main:create_app --factory`。
- 映像版本皆已固定；升級 Milvus 時請與 `uv.lock` 中的 pymilvus 版本相容（目前 pymilvus 2.6.17 對應 Milvus v2.6.25）；etcd、minio 沿用該版官方 standalone compose。
- LiteParse 為 Rust 原生套件，內含 pdfium 與 Tesseract，不需另外安裝 Node 或系統 Tesseract。預設語言（`chi_tra`、`chi_tra_vert`、`eng`，來自 tessdata_best 的固定 commit）已內建於 ingest 映像；`LITEPARSE_OCR_LANGUAGE` 改用其他語言時，該語言檔會在第一次 OCR 時下載（需要對外網路，且存在容器可寫層，重建映像後需重新下載）。
- 映像大小：ingest 與 web 都不含 torch，各約 1 GB；vllm 是官方映像（數 GB）。uv 的下載快取放在 BuildKit cache mount（`/root/.cache/uv`），不會進入映像層；`docker builder prune` 可清除它。程式碼以 `COPY --chown` 放入，不做遞迴 `chown`，避免複製整棵目錄樹。
- 根目錄 `.dockerignore` 是白名單（先 `*` 全部排除，再用 `!` 放行 build 需要的檔案）。Dockerfile 新增 `COPY` 來源時，必須同步在 `.dockerignore` 明確放行（`tests/test_config.py` 會檢查）。

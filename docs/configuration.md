# 環境變數

Docker 部署的設定放在 `docker/.env`（範本 `docker/.env.example`），本機開發的設定放在根目錄的 `.env`（範本 `.env.example`）。兩個檔案互不相干，都已被 git 忽略。同名的系統環境變數優先於 `.env`。

改了 `docker/.env` 之後要執行 `docker compose up -d` 才會套用，`restart` 不會重新讀取。

以下這些在 Docker 裡由 `docker-compose.yml` 固定，`docker/.env` 改了也沒有作用，只有本機開發時才需要設定：`DB_PATH`、`KB_DIR`、`MILVUS_URI`、`EMBEDDER_URL`、`OLLAMA_BASE_URL`、`KB_INGEST_MODE`、`OCR_BACKEND`、`VLLM_BASE_URL`。各服務的並行數與批次大小，以及選這些數字的原因，整理在 [services.md](services.md#並行數與批次大小)。

| 變數 | 預設值 | 說明 |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 位址 |
| `CHAT_MODEL` | `gemma4:e4b` | 聊天使用的模型（Ollama 預設值；可在「系統設定」頁覆寫） |
| `GEMINI_API_KEY` | 空 | Google Gemini API 金鑰；只能放在 `.env`／`docker/.env`，留空則「系統設定」中的 Gemini 選項停用 |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Gemini 預設模型（可在「系統設定」頁覆寫） |
| `LLM_PROVIDER` | `gemini` | 預設的聊天供應商（`gemini` 或 `ollama`），可在「系統設定」頁覆寫 |
| `OLLAMA_TIMEOUT_SECONDS` | `300` | Ollama 最多可以幾秒沒有回應（等第一段文字，或兩段之間）。CPU 上載入模型可能要幾分鐘；超過就停止這次回答，對話不會一直卡在「正在產生回覆」 |
| `OLLAMA_NUM_CTX` | `16384` | Ollama 每次對話的 context 長度。Ollama 本身預設只有 4096，知識庫回答帶整頁圖（每張約 740 token）時會被截斷 |
| `ADMIN_USERNAME` | 空 | 管理員帳號，啟動時自動建立（3–32 字元） |
| `ADMIN_PASSWORD` | 空 | 管理員密碼，每次啟動同步到資料庫（8 字元以上，最多 72 位元組） |
| `DB_PATH` | `<專案根目錄>/data/app.db` | SQLite 資料庫路徑 |
| `SESSION_DAYS` | `7` | 登入工作階段有效天數 |
| `ALLOW_REGISTRATION` | `false` | 是否開放任何人在登入頁註冊。關閉時只有管理員帳號（由 `.env` 建立）能登入，除非之前已經註冊過 |
| `COOKIE_SECURE` | `false` | 登入 cookie 只透過 HTTPS 傳送。放在 HTTPS 反向代理後面時設成 `true`；沒有 HTTPS 時開啟會無法登入 |
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
| `REPLY_LOCK_TTL_SECONDS` | `90` | 對話回覆鎖逾時秒數（串流中每 30 秒更新一次，必須大於 30） |
| `OCR_BACKEND` | `transformers` | `transformers` 或 `vllm` |
| `VLLM_BASE_URL` | `http://localhost:8001/v1` | vLLM 位址 |
| `OCR_VLLM_KV_CACHE_MEMORY` | `512M` | vLLM 的 KV cache 大小（僅 Docker，見 [gpu-memory.md](gpu-memory.md)） |
| `OCR_VLLM_GPU_UTIL` | `0.7` | vLLM 啟動時的顯存檢查比例，剩餘顯存不足就拒絕啟動（僅 Docker） |
| `OCR_VLLM_MAX_MODEL_LEN` | `18000` | vLLM 的最大上下文長度（僅 Docker） |
| `OCR_VLLM_MAX_NUM_SEQS` | `4` | vLLM 同時處理的請求上限（僅 Docker） |
| `EMBED_MODEL_ID` | `google/embeddinggemma-2` | embedder 載入的模型（僅 Docker，設在 `docker/.env`）；更換後需重建知識庫 |
| `EMBED_IMAGE_TOKENS` | `280` | embedder 每張圖片的 soft token 數，只接受 70、140、280、560、1120（僅 Docker） |
| `EMBED_TIMEOUT_SECONDS` | `180` | embedder 一次運算超過這個秒數就視為卡住，程序結束後由 Docker 重啟（僅 Docker） |
| `MILVUS_URI` | `http://localhost:19530` | Milvus 位址（快速測試：`cd docker && docker compose up -d milvus`；Docker 內固定為 `http://milvus:19530`） |
| `MILVUS_TOKEN` | 空 | Milvus 驗證 token（未啟用驗證則留空） |
| `MILVUS_COLLECTION` | `phoebe_chunks` | 存放知識庫段落的 collection 名稱 |
| `WEB_CONCURRENCY` | `2` | Gunicorn worker 數（僅 Docker，設在 `docker/.env`） |
| `PHOEBE_PORT` | `8000` | Docker 對外連接埠。預設所有網路介面都連得到；設成 `127.0.0.1:8000` 就只有本機能連，放在反向代理後面時要這樣設（見 [docker/README.md](../docker/README.md#對外開放與反向代理)） |
| `MONITOR_INTERVAL_SECONDS` | `5` | 監控頁面的資源取樣間隔（秒，monitor 服務） |
| `MONITOR_RESOURCE_RETENTION_HOURS` | `48` | 資源取樣保留幾小時 |
| `MONITOR_EVENT_RETENTION_DAYS` | `30` | 任務事件保留幾天 |
| `FRONTEND_DIST` | `<專案根目錄>/frontend/dist` | 前端建置輸出資料夾，由後端提供；使用相對路徑時，會以啟動 uvicorn 的工作目錄為基準 |

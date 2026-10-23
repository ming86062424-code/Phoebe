# 服務架構

`cd docker && docker compose up` 會啟動 7 個服務，另外還會用到兩個外部的聊天模型服務。這份文件說明每個服務的工作、它們怎麼互相呼叫，以及資料存在哪裡。設定與操作方式見 [docker/README.md](../docker/README.md)。

## 總覽

```
                 瀏覽器（http://localhost:8000）
                          │
                          ▼
   ┌───────────────────── web ─────────────────────┐
   │ 網頁介面與 API：登入、對話、知識庫、上傳、檢索    │
   └──┬──────────────┬──────────────┬──────────────┘
      │ SQLite 佇列    │ 查詢向量      │ 聊天
      ▼               ▼              ▼
   ingest ────────► embedder      Gemini（雲端）／主機 Ollama
      │  解析         ▲
      ├─► LiteParse（ingest 內）
      ├─► vllm（LightOnOCR，GPU）
      │  寫入向量      │
      ▼               │
   milvus ◄───────────┘（web 也直接查詢 milvus）
      ├─ etcd
      └─ minio
```

## Docker 裡的服務

| 服務 | 映像 | 功能 | 對外埠 | GPU |
|---|---|---|---|---|
| **web** | `phoebe-web:local` | 主程式（Gunicorn + FastAPI）。提供網頁介面與所有 API；提問時負責檢索、組合參考資料與附圖，再呼叫聊天模型並串流回應 | `8000`（`PHOEBE_PORT`） | 否 |
| **ingest** | `phoebe-ingest:local` | 背景處理上傳的文件：從佇列取出 → 解析 → 請 embedder 產生向量 → 寫入 Milvus。LiteParse 直接在這個容器裡執行 | 無 | 否 |
| **embedder** | `phoebe-embedder:local` | 向量產生服務，用 sentence-transformers 在 CPU 以 float32 執行 `google/embeddinggemma-2`（768 維）。問題、文字 chunk、整頁圖片的向量都由它產生 | 只限本機 `8002` | 否 |
| **vllm** | `vllm/vllm-openai:v0.30.0` | LightOnOCR 的 OCR 服務。ingest 選用 LightOnOCR 時，把每一頁的圖片送來辨識文字並標出圖片區塊（grounding） | 只限本機 `8001` | **是** |
| **milvus** | `milvusdb/milvus:v2.6.25` | 向量資料庫。所有知識庫共用 `phoebe_chunks` 這個 collection，負責向量查詢（cosine）與關鍵字查詢（BM25，jieba 斷詞） | 只限本機 `19530` | 否 |
| **etcd** | `quay.io/coreos/etcd:v3.5.25` | Milvus 的設定儲存（collection 結構等），只給 Milvus 使用 | 無 | 否 |
| **minio** | `milvusdb/minio:RELEASE.2024-12-18T13-15-44Z` | Milvus 的物件儲存（實際的資料檔），只給 Milvus 使用 | 無 | 否 |

「只限本機」表示只綁在 `127.0.0.1`，同網路的其他電腦連不到。這三個服務本身都沒有驗證機制。

### 啟動順序

- web 等 milvus、embedder 健康檢查通過後才啟動
- ingest 等 web、milvus、embedder 健康後才啟動
- vllm 只在啟用 `lightonocr` profile 時才會啟動。ingest 不等它健康：vLLM 載入模型要幾分鐘，這段期間 LiteParse 照常可用；沒啟用時 ingest 也照常啟動，LightOnOCR 顯示「目前無法使用」
- LightOnOCR 的文件如果在 vLLM 就緒前排入佇列，會留在「排隊中」，等 vLLM 好了再自動處理

## ingest：文件處理服務

ingest 負責把上傳的文件變成可以檢索的 chunk。它是知識庫唯一的「處理者」：web 只負責接收檔案、排入佇列，之後的工作都由 ingest 完成。

### 一份文件的處理流程

1. 從佇列（SQLite 的 `kb_documents` 表）認領一份「排隊中」的文件
2. 依文件類型與**知識庫設定的解析器**，呼叫對應的解析模組
3. 請 embedder 為每個 chunk 產生向量
4. 寫入 Milvus，文件狀態改為「完成」；失敗則記錄原因，可在知識庫頁重試

### 解析模組在哪裡執行

| 解析模組 | 用於 | 在哪裡執行 | ingest 做的事 |
|---|---|---|---|
| **LiteParse** | PDF、圖檔（知識庫設定為 LiteParse 時） | ingest 容器內，CPU | 全部：抽文字、Tesseract OCR、整頁圖、抽出內嵌圖片 |
| **LightOnOCR** | PDF、圖檔（知識庫設定為 LightOnOCR 時） | 辨識在 **vllm** 服務（GPU） | 把每頁轉成圖片送給 vLLM，收到結果後解析區塊、裁切圖片、組成 markdown |
| **文字解析** | txt、md（一律使用，不看知識庫設定） | ingest 容器內 | 全部：依段落或標題切塊 |

每個解析模組怎麼切 chunk、送什麼給 embedder，見 [parsers.md](parsers.md)。

新增解析模組時的原則：如果它只是一個 Python 套件（像 LiteParse），直接放進 ingest；如果它需要常駐的大型模型或 GPU（像 LightOnOCR），就交給獨立的模型服務，由 ingest 透過 HTTP 呼叫，並用 compose profile 讓它可以選擇是否啟動。

### 同時處理

| 設定 | 預設 | 說明 |
|---|---|---|
| `INGEST_CONCURRENCY` | 1 | ingest 同時處理幾份文件。一個分派者負責認領文件，最多交給這麼多條執行緒同時處理 |
| `OCR_CONCURRENCY` | 4 | LightOnOCR 處理一份文件時，同時送給 vLLM 的頁數 |
| `OCR_TIMEOUT_SECONDS` | 600 | 每一頁 OCR 請求的逾時秒數 |

vLLM 同時收到的請求數 = `INGEST_CONCURRENCY` × `OCR_CONCURRENCY`。這個乘積不要超過 vLLM 的 `--max-num-seqs`（預設 4，`OCR_VLLM_MAX_NUM_SEQS`），超過的部分會在 vLLM 內排隊、拖慢每一頁。

程式不限制這兩個值，換了更強的硬體時調高即可；目前的預設值是配合 RTX 2060 6 GB 與 16 GB 記憶體的 Docker 環境。實測數據見 [vllm-rtx2060.md](vllm-rtx2060.md)。

**調高 `INGEST_CONCURRENCY` 前先看記憶體**（2026-10-23 實測）：LiteParse 每處理一份 PDF 都要把頁面轉成 400 DPI 的圖、跑 Tesseract，同時處理 2 份時 ingest 的記憶體高峰約 **3.2 GB**（一次一份時約 0.2–0.4 GB），embedder 高峰約 5 GB。Docker 虛擬機只有 15.5 GB，在同時開著 vLLM 和其他程式（例如約 5 GB 的 RAGFlow）的情況下設成 2，虛擬機記憶體不足，系統終止了 embedder，處理中的文件因此失敗；關掉 vLLM 後同樣設定就能順利完成。

### 可靠性

- **解析模組暫時不能用**（例如 vLLM 還在載入）：該解析器的文件留在「排隊中」，其他文件照常處理，恢復後自動接續
- **心跳**：處理中的文件會定時更新時間；ingest 當掉而停住的文件，超過 `KB_STALE_SECONDS` 會被重新排入
- **文件在處理中被刪除**：處理到下一頁時發現並停止，不會寫入任何 chunk
- **執行緒安全**：PDF 轉圖用的 pdfium 不是執行緒安全的，所有 pypdfium2 的呼叫都經過同一把鎖；LiteParse 內建自己的 pdfium，不受影響

## Docker 外的服務

| 服務 | 功能 |
|---|---|
| **Gemini（雲端）** | 預設的聊天模型（`LLM_PROVIDER=gemini`）。需要在 `docker/.env` 設定 `GEMINI_API_KEY` |
| **主機上的 Ollama** | 選用的本機聊天模型（`gemma4:e4b`）。web 透過 `host.docker.internal:11434` 連線。vllm 開著時 GPU 會被佔走，Ollama 只能用 CPU 執行，見 [vllm-rtx2060.md](vllm-rtx2060.md) |

管理員可以在「系統設定」頁切換聊天的供應商與模型。

## 資料流程

### 上傳文件

1. **web** 檢查檔案大小、副檔名與內容，把原始檔存到 `docker/volumes/kb/`，在 SQLite 建立文件紀錄並排入佇列
2. **ingest** 取出文件，依類型解析：
   - **PDF／圖檔 + LiteParse**：每頁轉成整頁圖，並抽出文字和頁面中內嵌的圖片
   - **PDF／圖檔 + LightOnOCR**：每頁送到 **vllm** 辨識，得到 markdown 文字，並依座標裁出圖片
   - **txt／md**：直接切段，不需解析器
3. **ingest** 請 **embedder** 產生向量：
   - LiteParse 的頁面用「整頁圖 + 該頁文字」
   - 其他的只用文字
4. **ingest** 把每個 chunk（文字、向量、圖片清單）寫進 **milvus**，文件狀態改為「完成」

### 提問（選了知識庫的對話）

1. **web** 請 **embedder** 把問題轉成向量
2. **web** 在 **milvus** 同時做向量查詢與 BM25 關鍵字查詢，兩邊的分數正規化後加權相加（`KB_VECTOR_WEIGHT`，預設各半），取前 5 筆（`KB_TOP_K`）
3. **web** 把這 5 筆參考資料的文字放進系統提示詞，並依排名附上最多 4 張圖（`KB_MAX_IMAGES`）
4. **web** 呼叫 Gemini 或 Ollama，把回答和參考資料卡片串流回瀏覽器

## 資料存在哪裡

| 位置 | 內容 | 可以刪嗎 |
|---|---|---|
| 主機目錄 `docker/volumes/kb/` | 上傳的原始檔、整頁圖、裁切圖 | 刪了就要重新上傳 |
| volume `phoebe_phoebe-db` | SQLite：帳號、對話、知識庫與文件清單、系統設定 | **不可以**，所有帳號與對話都在這裡 |
| volume `phoebe_milvus-data`、`phoebe_etcd-data`、`phoebe_minio-data` | 向量資料庫 | 刪了可以用 `reprocess_all` 從原始檔重建（三個要一起刪） |
| volume `phoebe_vllm-hf-cache` | LightOnOCR 模型快取（約 1.6 GB） | 可以，下次啟動會重新下載 |
| volume `phoebe_vllm-triton-cache` | vLLM 的 Triton 編譯快取（約 180 MB） | 可以，下次啟動會重新編譯（多花幾分鐘） |
| volume `phoebe_embedder-hf-cache` | embeddinggemma-2 模型快取（約 1.5 GB） | 可以，下次啟動會重新下載 |

資料庫與快取放在 Docker 管理的 volume，上傳的檔案放在主機目錄，原因見 [docker-storage.md](docker-storage.md)。備份方式見 [docker/README.md](../docker/README.md) 的「資料備份」。

## 哪些服務可以不開

| 情境 | 做法 |
|---|---|
| 只用 LiteParse，想把 GPU 讓給 Ollama | vllm 預設不啟動；已啟動的話 `docker compose --profile lightonocr stop vllm`，之後要 OCR 時再 `docker compose --profile lightonocr up -d` |
| 其他服務 | 都是必要的：少了 embedder 或 milvus，上傳和檢索都會失敗；etcd、minio 是 Milvus 的一部分 |

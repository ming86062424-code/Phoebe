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
- ingest 等 web、milvus、embedder 健康後才啟動。vllm 只要容器開始執行即可，不必等健康：vLLM 載入模型要幾分鐘，這段期間 LiteParse 照常可用
- LightOnOCR 的文件如果在 vLLM 就緒前排入佇列，會留在「排隊中」，等 vLLM 好了再自動處理

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
| `docker/volumes/kb/` | 上傳的原始檔、整頁圖、裁切圖 | 刪了就要重新上傳 |
| `docker/volumes/milvus/`、`etcd/`、`minio/` | 向量資料庫 | 刪了可以用 `reprocess_all` 從原始檔重建 |
| `docker/volumes/vllm-hf-cache/` | LightOnOCR 模型快取（約 2 GB） | 可以，下次啟動會重新下載 |
| `docker/volumes/embedder-hf-cache/` | embeddinggemma-2 模型快取（約 1.5 GB） | 可以，下次啟動會重新下載 |
| Docker volume `phoebe_phoebe-db` | SQLite：帳號、對話、知識庫與文件清單、系統設定 | **不可以**，所有帳號與對話都在這裡 |

SQLite 刻意放在 Docker 管理的 volume，而不是 `docker/volumes/`：web 和 ingest 會同時讀寫它，Windows 目錄掛載的檔案鎖不可靠。備份方式見 [docker/README.md](../docker/README.md) 的「資料備份」。

## 哪些服務可以不開

| 情境 | 做法 |
|---|---|
| 只用 LiteParse，想把 GPU 讓給 Ollama | `docker compose stop vllm`；之後要 OCR 時再 `docker compose start vllm` |
| 其他服務 | 都是必要的：少了 embedder 或 milvus，上傳和檢索都會失敗；etcd、minio 是 Milvus 的一部分 |

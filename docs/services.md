# 服務架構

`cd docker && docker compose up` 會啟動 7 個服務（加上 `--profile lightonocr` 時再多一個 vllm，共 8 個），另外還會用到兩個外部的聊天模型服務。這份文件說明每個服務的工作、它們怎麼互相呼叫，以及資料存在哪裡。設定與操作方式見 [docker/README.md](../docker/README.md)；完整的架構圖，以及上傳文件和對話的時序圖見 [architecture.md](architecture.md)。

## 總覽

服務之間怎麼連接，見 [architecture.md](architecture.md) 的架構圖。簡單來說：瀏覽器只連 web；web 處理對話和檢索，把上傳的文件放進 SQLite 的佇列；ingest 從佇列取出文件，解析後請 embedder 算向量，寫進 Milvus；monitor 在背景記錄資源用量。

## Docker 裡的服務

| 服務 | 映像 | 功能 | 對外埠 | GPU |
|---|---|---|---|---|
| **web** | `phoebe-web:local` | 主程式（Gunicorn + FastAPI）。提供網頁介面與所有 API；提問時負責檢索、組合參考資料與附圖，再呼叫聊天模型並串流回應 | `8000`（`PHOEBE_PORT`） | 否 |
| **ingest** | `phoebe-ingest:local` | 背景處理上傳的文件：從佇列取出 → 解析 → 請 embedder 產生向量 → 寫入 Milvus。LiteParse 直接在這個容器裡執行 | 無 | 否 |
| **embedder** | `phoebe-embedder:local` | 向量產生服務，用 sentence-transformers 以 float32 執行 `google/embeddinggemma-2`（768 維，不載入用不到的音訊編碼器）。問題、文字 chunk、整頁圖片的向量都由它產生 | 只限本機 `8002` | 預設否；不用 LightOnOCR 時可改用 GPU（[gpu-memory.md](gpu-memory.md)） |
| **vllm** | `vllm/vllm-openai:v0.30.0` | LightOnOCR 的 OCR 服務。ingest 選用 LightOnOCR 時，把每一頁的圖片送來辨識文字並標出圖片區塊（grounding） | 只限本機 `8001` | **是** |
| **milvus** | `milvusdb/milvus:v2.6.25` | 向量資料庫。所有知識庫共用 `phoebe_chunks` 這個 collection，負責向量查詢（cosine）與關鍵字查詢（BM25，jieba 斷詞） | 只限本機 `19530` | 否 |
| **etcd** | `quay.io/coreos/etcd:v3.5.25` | Milvus 的設定儲存（collection 結構等），只給 Milvus 使用 | 無 | 否 |
| **minio** | `milvusdb/minio:RELEASE.2024-12-18T13-15-44Z` | Milvus 的物件儲存（實際的資料檔），只給 Milvus 使用 | 無 | 否 |
| **monitor** | `phoebe-web:local` | 監控頁面的資源取樣：每 5 秒記錄所有容器的記憶體、CPU 與 GPU 用量，寫進 `monitor.db`。需要讀取 Docker socket，見 [monitoring.md](monitoring.md) | 無 | 只讀取用量（`nvidia-smi`） |

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

ingest 同時處理幾份文件、LightOnOCR 一次送幾頁，以及這些數字的原因，見下方的[並行數與批次大小](#並行數與批次大小)。

### 可靠性

- **解析模組暫時不能用**（例如 vLLM 還在載入）：該解析器的文件留在「排隊中」，其他文件照常處理，恢復後自動接續
- **embedder 暫時連不上**（例如記憶體不足被系統終止、正在重新載入模型）：建立索引時會等 10、30、60 秒各重試一次，共重試 3 次；都失敗才把文件標為失敗（「無法連線到 embedding 服務，已重試 3 次」），之後可在知識庫頁按「重試」。提問時的查詢向量不重試，直接回報錯誤，避免回答卡住好幾分鐘
- **embedder 卡住**：一次運算超過 `EMBED_TIMEOUT_SECONDS`（預設 180 秒）就視為卡住，embedder 自行結束，由 Docker 重啟；正在建立索引的文件會照上一點的方式重試。Docker 本身不會重啟「卡住但沒結束」的服務，所以這個判斷做在 embedder 裡
- **心跳**：處理中的文件會定時更新時間；ingest 當掉而停住的文件，超過 `KB_STALE_SECONDS` 會被重新排入
- **文件在處理中被刪除**：處理到下一頁時發現並停止，不會寫入任何 chunk
- **執行緒安全**：PDF 轉圖用的 pdfium 不是執行緒安全的，所有 pypdfium2 的呼叫都經過同一把鎖；LiteParse 內建自己的 pdfium，不受影響

## Docker 外的服務

| 服務 | 功能 |
|---|---|
| **Gemini（雲端）** | 預設的聊天模型（`LLM_PROVIDER=gemini`）。需要在 `docker/.env` 設定 `GEMINI_API_KEY` |
| **主機上的 Ollama** | 選用的本機聊天模型（`gemma4:e4b`）。web 透過 `host.docker.internal:11434` 連線。vllm 開著時 GPU 會被佔走，Ollama 只能用 CPU 執行，見 [vllm-rtx2060.md](vllm-rtx2060.md) |

管理員可以在「系統設定」頁切換聊天的供應商與模型。

## 並行數與批次大小

各服務「同時處理幾個」「一次送幾個」的數字都整理在這裡。預設值是配合開發機設定的：RTX 2060 6 GB 顯卡，Docker 虛擬機約 16 GB 記憶體。換了硬體可以調整「可設定」那一欄有列出變數的項目。

### 總表

| 服務 | 什麼東西的數量 | 預設 | 可設定 |
|---|---|---|---|
| web | Gunicorn worker（程序）數 | 2 | `WEB_CONCURRENCY` |
| web | 每個 worker 處理請求的主 thread | 1 | 否 |
| web | 同一個對話同時產生的回覆數 | 1 | 否 |
| ingest | 同時處理的文件數 | 1 | `INGEST_CONCURRENCY` |
| ingest | 多久查一次佇列 | 2 秒 | `KB_INGEST_POLL_SECONDS` |
| ingest | LightOnOCR：一份文件同時送給 vLLM 的頁數 | 4 | `OCR_CONCURRENCY` |
| ingest | 用 pypdfium2 處理 PDF（LightOnOCR 轉頁面圖、LiteParse 讀總頁數） | 一次 1 個呼叫 | 否 |
| vllm | 同時處理的頁數（序列數） | 4 | `OCR_VLLM_MAX_NUM_SEQS` |
| embedder | 程序數 | 1 | 否 |
| embedder | 同時運算的請求數，其他請求排隊 | 1 | 否 |
| embedder | 每個請求最多幾段文字 | 16 段 | 否（`app/kb/embedder_client.py` 的 `BATCH_SIZE`） |
| embedder | 每個請求最多幾張整頁圖（各連同該頁文字） | 2 張 | 否（`app/kb/index.py` 的 `IMAGE_BATCH_SIZE`） |
| embedder | 一次運算超過多久算卡住 | 180 秒 | `EMBED_TIMEOUT_SECONDS` |
| 回答時 | 附給聊天模型的圖片上限（張數，不是並行數） | 4 張 | `KB_MAX_IMAGES` |

Milvus、Gemini、Ollama 的並行上限 Phoebe 沒有設定，由它們自己決定。Gemini 另外受每分鐘請求數的額度限制。

### 要一起看的關係

- **vLLM 同時收到的頁數 = `INGEST_CONCURRENCY` × `OCR_CONCURRENCY`**。這個乘積不要超過 `OCR_VLLM_MAX_NUM_SEQS`，超過的部分會在 vLLM 裡排隊，拖慢每一頁。三個值要一起調。
- **embedder 是全系統共用的瓶頸**。web 的「問題轉向量」和 ingest 的「建立索引」排同一個隊，所以文件正在建立索引時，聊天開始回應會變慢。

### 為什麼是這些數字

#### web

- **worker 數 2**：當初沒有留下選 2 的紀錄。推論的理由：
  - 每個 worker 用 async 輪流處理請求，等聊天模型回應時不佔 CPU，一個 worker 就能同時服務很多對話，不需要開很多個。
  - 開 2 個，其中一個重啟或卡住時，另一個還能接請求。
  - 每多一個 worker 就多佔一份記憶體。
  - 實際能服務多少人，要用壓力測試（[loadtest.md](loadtest.md)）量。
- **每個 worker 1 個主 thread**：這是 async 的運作方式，不是設定出來的。主 thread 在等待時就去處理別的請求；會卡住主 thread 的工作，例如讀寫 SQLite、檢索，交給輔助 thread 做。唯一的例外是使用者中途離開、回答被取消時的收尾（釋放回覆鎖、記錄監控事件）：這時不能再等待，否則收尾本身也可能被取消，所以直接在主 thread 上執行。
- **同一個對話一次只產生 1 份回覆**：避免同一個對話裡兩份回覆同時寫入，例如連按兩次送出，或兩個分頁同時送。第二個請求會收到 409「這個對話正在產生回覆，請稍候」。鎖存在 SQLite，2 個 worker 都看得到；處理中的程序當掉時，鎖在 `REPLY_LOCK_TTL_SECONDS`（90 秒）內失效。

#### ingest 與 vLLM

- **同時處理 1 份文件**：記憶體不夠，不是速度。
  - LiteParse 每處理一份 PDF，都要把頁面轉成 400 DPI 的圖、跑 Tesseract，同時處理的份數越多，ingest 用的記憶體越多。2026-10-23 設成 2 份時，ingest 最高約 3.2 GB（同條件下只處理 1 份的用量還沒量過）。
  - 那次 Docker 虛擬機只有 15.5 GB，又同時開著 vLLM 和其他程式（例如約 5 GB 的 RAGFlow），記憶體不夠，系統終止了 embedder，處理中的文件因此失敗；關掉 vLLM 後，同樣的設定可以順利完成。
  - 當時 embedder 最高用到約 5 GB，之後已經降低：2026-10-24 起不載入用不到的音訊編碼器，2026-10-27 起整頁圖每次只送 2 張。現在 CPU 模式待機約 2.5 GB，處理一份 12 頁 PDF 時最高約 2.7 GB（[parser-benchmarks.md](parser-benchmarks.md)）。同時處理 2 份的情況還沒有重新量過。
  - 記憶體較大的機器可以調高，調高前先用監控頁看實際用量。
- **每 2 秒查一次佇列**：當初沒有留下紀錄。推論的理由：web 和 ingest 在不同容器，web 沒辦法直接通知 ingest，只能讓 ingest 定時查；文件處理動輒幾十秒以上，多等最多 2 秒影響不大，而查一次佇列只是一個很輕的 SQLite 查詢。
- **LightOnOCR 一次送 4 頁**：2026-10-23 在 RTX 2060 上實測，一份 12 頁的 PDF，一次 1 頁要 630 秒、2 頁 365 秒、4 頁 279 秒，4 頁最快，ingest 的記憶體只多約 150 MB（[parser-benchmarks.md](parser-benchmarks.md)、[vllm-rtx2060.md](vllm-rtx2060.md)）。
- **vLLM 最多同時 4 頁**：vLLM 預設最多 256 條序列，在 6 GB 顯卡上配不到 KV cache，無法啟動。改成 4，剛好配合「一次送 4 頁」；每頁約需 7,000 個 token，512 MiB 的 KV cache 放得下 4 頁（[gpu-memory.md](gpu-memory.md)）。
- **pypdfium2 一次只處理 1 個呼叫**：pdfium 不是執行緒安全的，多個 thread 同時呼叫可能出錯，所以所有 pypdfium2 的呼叫都經過同一把鎖。LiteParse 用的是自己內建的 pdfium，不受這把鎖限制。

#### embedder

- **只有 1 個程序**：每個程序都要載入一份模型，CPU 模式約 2.5 GB。多開一個程序就多一份記憶體，在這台機器上不划算。
- **一次只算 1 個請求**：模型不是設計來同時被多個 thread 呼叫的，而且同時算多個請求，記憶體高峰會疊加。所以用一把鎖讓請求排隊，一次算一個。
- **文字每批 16 段**：當初沒有留下選 16 的紀錄。推論的理由：文字很輕，一批 16 段的記憶體負擔很小，又能把一份文件的上百段減少成少數幾次請求。
- **整頁圖每批 2 張**：圖片很吃記憶體，embedder 會把一個請求裡的圖一次算完，所以一批的張數就決定記憶體高峰。用同樣 8 張圖實測（2026-10-27 起改用 2 張）：一次 8 張時顯存高峰 3.12 GiB、2 張時 2.01 GiB，時間只多約 6%。CPU 模式的高峰也從約 3.5 GB 降到約 2.7 GB，建立索引沒有變慢（[gpu-memory.md](gpu-memory.md)）。
- **超過 180 秒算卡住**：一個請求最多是 2 張整頁圖，CPU 上大約十幾秒到幾十秒，180 秒有十倍左右的餘裕，不會誤判正常的運算。卡住時 embedder 會自己結束，由 Docker 重啟；Docker 本身不會重啟「卡住但沒結束」的服務，所以這個判斷做在 embedder 裡（2026-11-03 加入）。

#### 回答時附圖上限 4 張

當初的設計規格只寫了「最多 4 張」，沒有寫原因。已知的考量：

- 每張圖都佔模型的上下文空間。用 Ollama 時，一張整頁圖約 740 個 token，圖多了可能超過上下文長度，回答會被截斷，所以 Ollama 的上下文預設調到 16384（`OLLAMA_NUM_CTX`）。
- 圖片和參考資料是靠順序和文字說明對應的，圖越多，模型越容易對錯。
- 圖越多，送給模型的資料越大，回答越慢，雲端模型的費用也越高。

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
| volume `phoebe_phoebe-db` | SQLite：帳號、對話、知識庫與文件清單、系統設定（`app.db`）；監控資料（`monitor.db`） | `app.db` **不可以**，所有帳號與對話都在這裡；`monitor.db` 可以，見 [monitoring.md](monitoring.md) |
| volume `phoebe_milvus-data`、`phoebe_etcd-data`、`phoebe_minio-data` | 向量資料庫 | 刪了可以用 `reprocess_all` 從原始檔重建（三個要一起刪） |
| volume `phoebe_vllm-hf-cache` | LightOnOCR 模型快取（約 1.6 GB） | 可以，下次啟動會重新下載 |
| volume `phoebe_vllm-triton-cache` | vLLM 的 Triton 編譯快取（約 180 MB） | 可以，下次啟動會重新編譯（多花幾分鐘） |
| volume `phoebe_embedder-hf-cache` | embeddinggemma-2 模型快取（約 1.5 GB） | 可以，下次啟動會重新下載 |

資料庫與快取放在 Docker 管理的 volume，上傳的檔案放在主機目錄，原因見 [docker-storage.md](docker-storage.md)。備份方式見 [docker/README.md](../docker/README.md) 的「資料備份」。

## 哪些服務可以不開

| 情境 | 做法 |
|---|---|
| 只用 LiteParse，想把 GPU 讓給 Ollama | vllm 預設不啟動；已啟動的話 `docker compose --profile lightonocr stop vllm`，之後要 OCR 時再 `docker compose --profile lightonocr up -d` |
| 只用 LiteParse，想讓建立索引更快 | 不開 vllm，改用 GPU 版的 embedder：`docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build`（12 頁的建立索引從約 100 秒降到 7 秒），見 [gpu-memory.md](gpu-memory.md) |
| 其他服務 | 都是必要的：少了 embedder 或 milvus，上傳和檢索都會失敗；etcd、minio 是 Milvus 的一部分 |

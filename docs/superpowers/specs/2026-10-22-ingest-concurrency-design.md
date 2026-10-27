# ingest 服務：改名、並行處理與 vLLM profile — 設計規格

- 日期：2026-10-22
- 分支：`feature/ingest`（從 `develop`），完成後合併回 `develop`
- 狀態：**待使用者確認 spec 與開發計畫後才開始開發**

> **之後的變更**（2026-10-27 補註）：以下內容是當時的設計，保留作為紀錄。後來改變的部分：
>
> - 已實作完成（狀態「待使用者確認」是撰寫當時的狀態）。`OCR_CONCURRENCY` 實測後預設為 4（`docs/vllm-rtx2060.md`）。

## 1. 目的與範圍

**需求（使用者確認）**

1. 背景處理文件的服務 `worker` 全面改名為 `ingest`，包含程式、設定、Docker、文件。Gunicorn 的 worker 是 Gunicorn 自己的術語，不改。
2. `docs/services.md` 寫清楚 ingest 的職責，包括它如何依知識庫設定呼叫各個解析模組、每個模組在哪裡執行。
3. ingest 能**同時處理多份文件**，數量由 `INGEST_CONCURRENCY` 設定，預設 1。用單一程序內的多條執行緒實作，不多開容器。
4. LightOnOCR **一次送多頁給 vLLM**，數量由 `OCR_CONCURRENCY` 設定。用 1、2、4 頁實測速度後決定預設值。
5. vLLM 放進 compose 的 `lightonocr` profile，**預設不啟動**，要用時在指令加 `--profile lightonocr`。
6. vLLM 的 `--max-num-seqs` 從 8 改為 4。

**原則（使用者確認）**：並行機制要完整做出來，不因目前硬體跑不動就限制程式能力；實際使用多少，由設定配合硬體決定。

**明確不做**：
- 提問與文件處理的 embedding 分流
- 佇列公平排程
- 多開 ingest 容器
- 限制容器記憶體
- 改用 PostgreSQL 或專門的佇列工具

## 2. 改名：worker → ingest

| 類別 | 舊 | 新 |
|---|---|---|
| compose 服務 | `worker` | `ingest` |
| 映像 | `phoebe-worker:local` | `phoebe-ingest:local` |
| Dockerfile 建置目標 | `worker` | `ingest` |
| 程式模組 | `app/kb/worker.py`、`app/kb/worker_main.py` | `app/kb/ingest.py`、`app/kb/ingest_main.py` |
| 類別 | `IngestWorker` | `Ingester` |
| `KbServices` 屬性 | `.worker` | `.ingester` |
| 設定欄位 | `kb_worker_mode`、`kb_worker_poll_seconds` | `kb_ingest_mode`、`kb_ingest_poll_seconds` |
| 環境變數 | `KB_WORKER_MODE`、`KB_WORKER_POLL_SECONDS` | `KB_INGEST_MODE`、`KB_INGEST_POLL_SECONDS` |
| 測試檔 | `tests/test_kb_worker.py` | `tests/test_ingest.py` |
| 使用者看得到的訊息 | 「文件解析服務（worker）未啟動」 | 「文件處理服務（ingest）未啟動」 |

另外：
- 執行緒名稱、log 訊息、註解、文件、`reprocess_all` 的說明和指令都同步修改，例如 `docker compose run --rm --no-deps ingest python -m app.kb.reprocess_all`。
- 不是指 ingest 的「worker」改寫成更明確的說法：
  - 指 Gunicorn 程序的（`app/deps.py`、`app/chat_stream.py`）改為「Gunicorn 程序」。
  - 指 Starlette 執行緒池的（`app/routes/kb.py`）改為「背景執行緒」。
- **不改**：`docker/gunicorn.conf.py` 的 `workers`／`worker_class`、`WEB_CONCURRENCY`、`uvicorn-worker` 套件。

**舊環境變數**：設定了 `KB_WORKER_MODE` 或 `KB_WORKER_POLL_SECONDS`（非空值）時，`Settings.from_env()` 拒絕啟動，錯誤訊息說明新名稱，例如「KB_WORKER_POLL_SECONDS 已改名為 KB_INGEST_POLL_SECONDS」。做法和 `ADMIN_USERNAMES` 相同。

## 3. ingest 同時處理多份文件

### 3.1 結構：一個分派者 ＋ 執行緒池

```
分派者（1 條執行緒）
  ├─ 正在處理的數量 < INGEST_CONCURRENCY → 認領 1 份文件，交給執行緒池
  ├─ 已滿 → 等任何一份處理完（被喚醒）再繼續
  └─ 沒有可認領的文件，且正在處理的數量 = 0 → 釋放解析器資源、做定期維護

執行緒池（ThreadPoolExecutor，max_workers = INGEST_CONCURRENCY）
  └─ 每條執行既有的 process(doc_id)，結束時正在處理數 −1 並喚醒分派者
```

- **只有分派者會認領文件**：「目前在等哪個解析器」這類狀態只有它讀寫，不用加鎖。
- **只有分派者做定期維護**：重排卡住的文件、清理孤立檔案，不會 N 份同時跑。
- **釋放解析器資源**（例如卸載本機 OCR 模型）只在正在處理數為 0、而且沒有文件可認領時進行，不會卸載別人正在用的模型。
- **不變的部分**：
  - `process()` 每份文件各開自己的 SQLite 連線，各有一條心跳執行緒。
  - 認領本來就是原子性的；文件被刪除、被取代、停止服務時的處理也維持原樣。
- **停止服務**：分派者停止認領 → 正在處理的文件照原機制在下一頁停下（`_Stopped`）→ 等執行緒池結束。沒做完的文件會在下次啟動時被重排。
- **不開執行緒模式**（`threaded=False`，單元測試用）：維持依序處理，原有測試不變。

### 3.2 共用資源的執行緒安全

| 資源 | 處理方式 |
|---|---|
| **pypdfium2（pdfium）** | pdfium 不是執行緒安全的：同一程序內，即使是不同文件，也不能同時呼叫。新增一把全程序共用的鎖（`app/kb/parsers/pdfium_lock.py`），所有 pypdfium2 呼叫都在鎖內進行：開檔、讀頁數、轉圖、關檔 |
| **LiteParse** | 內部自帶 pdfium。開發時用壓力測試確認多條執行緒同時解析是否安全（同時解析多份 PDF，比對結果與依序解析相同、不當掉）。不安全的話，LiteParse 的解析與截圖也要用同一套鎖排隊 |
| EmbedderClient（httpx.Client） | httpx 的 Client 可以跨執行緒共用。開發時確認 |
| MilvusClient（gRPC） | gRPC 通道可以跨執行緒共用。開發時確認 |
| `KnowledgeIndex` | 建立 collection 的初始化已經有鎖 |
| 本機 transformers 版 OCR 引擎 | 共用一份 GPU 模型，`ocr()` 內加鎖，一次只讓一條執行緒使用 |

### 3.3 設定

- `INGEST_CONCURRENCY`：同時處理幾份文件，正整數，預設 **1**（行為與現在相同）。
- 嵌入模式（`KB_INGEST_MODE=embedded`，本機開發時在 web 程序內執行）使用同一個設定。

## 4. LightOnOCR 一次送多頁

### 4.1 流程

```
解析執行緒：依序轉頁面圖（pdfium 鎖內）→ 送進 OCR 執行緒池
            同時在途的頁數最多 OCR_CONCURRENCY；滿了就等最早的一頁完成
OCR 執行緒池：呼叫 engine.ocr(image)（HTTP 送 vLLM）
解析執行緒：收結果 → 解析 grounding、裁圖、存檔 → 回報進度 → 繼續送下一頁
```

- **頁面圖不會一次全部轉好放在記憶體**：同時在記憶體裡的頁面圖最多 `OCR_CONCURRENCY` 張。
- **進度回報、裁圖存檔、`check_current` 都留在解析執行緒做**：資料庫寫入維持單一執行緒，裁圖檔名依頁碼命名，彼此不衝突。
- **結果依頁碼排序**，和現在依序處理的結果相同。
- **任何一頁失敗**：取消尚未開始的頁、等在途的頁結束、捨棄結果，拋出「第 N 頁 OCR 失敗」。
- **文件被刪除或服務停止**（`on_progress` 拋出例外）：同上處理，再把例外原樣拋出。
- **圖檔**（單頁）不受影響。

### 4.2 設定與逾時

- `OCR_CONCURRENCY`：每份文件同時送給 vLLM 的頁數，正整數。預設值依 §6 的實測結果決定，範圍 1–4。
- **每頁請求的逾時改為設定 `OCR_TIMEOUT_SECONDS`**，預設 600，原本寫死 300。理由：多頁同時在途時，vLLM 的同時請求上限為 4，超過的請求會在 vLLM 內排隊，排隊時間也算在逾時內。
- vLLM 同時收到的請求數 = `INGEST_CONCURRENCY` × `OCR_CONCURRENCY`。文件說明這個乘積不要超過 vLLM 的 `--max-num-seqs`（4），超過的部分會排隊、拖長每頁的時間。

## 5. vLLM profile

- `vllm` 服務加上 `profiles: ["lightonocr"]`。
- `ingest` 對 `vllm` 的相依改為 `condition: service_started` 加上 `required: false`（Compose 2.20 以上支援，本機為 2.29.2）。vLLM 沒啟動時 ingest 照常啟動，LightOnOCR 顯示「目前無法使用」。
- `--max-num-seqs` 預設值改為 4：`${OCR_VLLM_MAX_NUM_SEQS:-4}`，`docker/.env.example` 同步修改。
- 指令：

  ```
  docker compose up -d                         # 不含 vLLM（只用 LiteParse）
  docker compose --profile lightonocr up -d    # 含 vLLM
  docker compose --profile lightonocr down     # 停止時也要帶 profile，否則 vLLM 不會停
  ```

- `docs/vllm-rtx2060.md` 的設定值與說明同步改為 4，並寫入 §6 的實測結果。

## 6. 實測（OCR_CONCURRENCY 的預設值）

- **環境**：Docker，`--max-num-seqs 4`，`INGEST_CONCURRENCY=1`，RAGFlow 等其他程式關閉以減少記憶體干擾。
- **檔案**：`test_data/草莓園入園方案.pdf`（12 頁），解析器 LightOnOCR。
- **量測**：`OCR_CONCURRENCY` = 1、2、4，各跑一次，記錄總處理時間、每頁平均時間，以及 ingest、vLLM、embedder 的記憶體高峰。
- **選擇規則**：取總時間最短、而且記憶體沒有逼近上限的值。結果寫進 `docs/vllm-rtx2060.md`。

## 7. 錯誤處理

| 情境 | 行為 |
|---|---|
| 設了舊的 `KB_WORKER_*` | 拒絕啟動，提示新名稱 |
| `INGEST_CONCURRENCY`、`OCR_CONCURRENCY`、`OCR_TIMEOUT_SECONDS` 不是正數 | 拒絕啟動，訊息說明 |
| 沒啟動 vLLM 卻選了 LightOnOCR | 和現在一樣：上傳時提示無法使用；已在佇列的文件留在佇列等待 |
| 某一頁 OCR 失敗或逾時 | 文件標為失敗：「第 N 頁 OCR 失敗：…」，可重試 |

## 8. 測試

- **改名**：
  - 舊環境變數被拒絕並提示新名稱。
  - compose、Dockerfile、`.dockerignore` 的設定測試改用新名稱。
  - 全部既有測試改名後通過。
- **並行處理**：
  - 用會「停住直到兩份都開始才放行」的假解析器，證明兩份文件同時進行（依序處理時測試會逾時失敗）。
  - 同時處理的數量不超過 `INGEST_CONCURRENCY`。
  - 停止服務時，正在處理的文件在下一頁停下，之後可被重排。
  - 只有全部閒下來才釋放解析器資源。
  - `INGEST_CONCURRENCY=1` 時行為與現在相同。
- **pdfium 鎖**：多條執行緒同時解析多份 PDF，結果與依序解析相同。
- **LiteParse**：多執行緒壓力測試，用來決定要不要加鎖。
- **一次送多頁**：
  - 假 OCR 引擎記錄同時在途的最大數量，等於 `OCR_CONCURRENCY`、不會超過。
  - 結果依頁碼排序。
  - 某頁失敗時，整份失敗並且不留下部分結果。
  - 文件被刪除時能停下。
  - 進度回報正確。
- **profile**：
  - compose 設定測試：`vllm` 在 `lightonocr` profile；ingest 的相依為 `required: false`；`--max-num-seqs` 預設 4。
- **Docker 實測**：
  - 不帶 profile 啟動：LiteParse 正常，LightOnOCR 顯示無法使用。
  - 帶 profile 啟動：LightOnOCR 正常。
  - `INGEST_CONCURRENCY=2` 時，兩份文件同時處理。
  - §6 的速度實測。

## 9. 升級注意

部署者需要做的事：
1. `docker/.env`：
   - `KB_WORKER_POLL_SECONDS` 改為 `KB_INGEST_POLL_SECONDS`，否則服務會拒絕啟動。
   - `OCR_VLLM_MAX_NUM_SEQS=8` 改為 4，或刪除這行改用新的預設值。
2. 第一次啟動新版時加上 `--remove-orphans`，清掉舊的 `worker` 容器，否則它會繼續搶同一個佇列：

   ```
   docker compose --profile lightonocr up -d --build --remove-orphans
   ```

3. 之後要用 LightOnOCR，都要在指令加上 `--profile lightonocr`。

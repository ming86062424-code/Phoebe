# 監控頁面：任務耗時與資源使用

日期：2026-10-25

> **之後的變更**（2026-10-27 補註）：以下內容是當時的設計，保留作為紀錄。後來改變的部分：
>
> - 已實作完成。monitor 不是以 root 執行：改為以 web 映像的一般使用者（UID 1000）執行，只額外加入 Docker socket 所屬的群組（`group_add`）。
> - 圖表分頁每 15 秒刷新，事件與資源分頁每 10 秒。

## 目標

管理員可以在獨立的「監控」頁面，確認每個任務花了多少時間，以及各時間點的資源用量。頁面分三個分頁：

1. **事件紀錄**：任務觸發的紀錄，包括文件處理、AI 對話、知識庫搜尋測試。每筆有狀態、總時間、各階段耗時。
2. **資源使用**：各容器與 GPU 目前的用量，以及詳細的取樣紀錄表。
3. **圖表**：選一段時間，用折線圖顯示記憶體、CPU、GPU 的用量，同時把這段時間內的任務標在時間軸上，方便對照「哪個任務造成高峰」。

LLM 的詳細追蹤（token 數、prompt 內容等）之後改用 Langfuse，**這次不做**。對話只記錄時間，不記錄訊息內容。

## 記錄哪些任務

| 類型 | 由誰記錄 | 階段 | 其他資訊 |
|---|---|---|---|
| 文件處理 `document` | ingest | 解析、向量（embedding）、寫入（Milvus） | 知識庫、文件、檔名、解析器、頁數、chunk 數、錯誤 |
| AI 對話 `chat` | web | 查詢向量、檢索、載入附圖、第一個字（從呼叫 LLM 到收到第一段回應，含思考）、生成 | 使用者、對話、知識庫、供應商、模型、參考資料數、附圖數、錯誤 |
| 知識庫搜尋測試 `search` | web | 查詢向量、檢索 | 知識庫、結果數、向量權重 |

狀態：`ok`、`error`、`cancelled`（對話中途中斷），文件另有 `stopped`（ingest 停止時留待下次）與 `deleted`（處理中被刪除）。

記錄失敗只寫 log，**絕不影響任務本身**。

## 資源取樣

- 新增 `monitor` 服務（使用 web 的映像，`python -m app.monitor.collector_main`），每 `MONITOR_INTERVAL_SECONDS`（預設 5）秒取樣一次：
  - **所有容器**（含其他專案，例如 RAGFlow，因為它們共用 Docker 虛擬機的記憶體）：透過 Docker Engine API（`/var/run/docker.sock`，唯讀掛載）取得記憶體、記憶體上限、CPU 使用率。記憶體算法與 `docker stats` 相同（扣掉 inactive file cache）。CPU 用前後兩次取樣的差值計算。
  - **GPU**：`nvidia-smi` 的顯存用量、總量、使用率。
- 安全：Docker socket 等同 Docker 的完整控制權。monitor 只呼叫讀取用的 API，不對外開任何 port；文件寫明這個風險，以及不需要時可以不啟動 monitor。socket 在 Docker Desktop 上屬於 root，所以 monitor 以 root 執行。
- 沒有 NVIDIA GPU 的機器：monitor 的 GPU 需求會讓它無法啟動，文件說明把 `deploy` 那段註解掉即可（只是少了 GPU 資料）。
- 本機開發模式（不在 Docker 執行 monitor）：事件照樣記錄，資源分頁顯示「monitor 服務未執行」。

## 儲存

- 另一個 SQLite 檔 `monitor.db`，和 `app.db` 放在同一個目錄（Docker 裡是 `phoebe-db` volume）。分開存放的原因：取樣頻繁寫入，不和帳號、對話的資料庫搶寫入；不需要時可以整個刪掉。
- 保留期限：取樣 `MONITOR_RESOURCE_RETENTION_HOURS`（預設 48 小時），事件 `MONITOR_EVENT_RETENTION_DAYS`（預設 30 天），由 monitor 定期清除。
- 時間一律存 epoch 秒，前端顯示成當地時間。

## API（只限管理員）

- `GET /api/monitor/events?kind=&limit=&before=`：事件列表，新的在前。
- `GET /api/monitor/status`：monitor 最後一次取樣的時間、錯誤訊息、取樣間隔，以及每個來源的最新數值。
- `GET /api/monitor/series?minutes=`：圖表資料。每個來源最多約 360 個點，同一區間取最大值（高峰最重要）。同時回傳這段時間內的事件。
- `GET /api/monitor/samples?minutes=&source=&limit=`：原始取樣列表，新的在前。

## 前端

- 側邊欄新增「監控」（只限管理員），路徑 `#/monitor`，三個分頁。
- 圖表自己用 SVG 畫，不加新套件。
- 事件與目前用量每 10 秒自動刷新。

## 驗證

- 單元測試：事件記錄（含失敗不影響任務）、Docker stats 的記憶體與 CPU 計算、取樣與清除、API 權限與分桶、前端三個分頁。
- 實測：在 Docker 跑起來，上傳文件、對話、搜尋，確認事件與資源都正確出現。

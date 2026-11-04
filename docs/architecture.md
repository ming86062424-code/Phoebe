# 系統架構圖與時序圖

用 Docker Compose 部署時的完整架構，以及「上傳文件到知識庫」和「對話」兩個流程的時序圖。各服務的詳細說明見 [services.md](services.md)，設定見 [docker/README.md](../docker/README.md)。

分層架構圖是 SVG（[images/architecture.svg](images/architecture.svg)，用 `uv run python -m scripts.draw_architecture` 產生，要改圖就改這支腳本再重跑）。其餘的圖用 Mermaid 寫成。兩種圖 GitHub 和 VS Code 的 Markdown 預覽都能直接顯示。

## 架構圖

![Phoebe 系統架構圖](images/architecture.svg)

### 連線細節

每條連線讀寫什麼：

```mermaid
flowchart LR
    user["使用者<br/>瀏覽器"]

    subgraph docker["Docker Compose"]
        web["<b>web</b><br/>Gunicorn（2 個 worker）+ FastAPI<br/>網頁介面（React）與所有 API<br/>檢索、組提示詞、串流回答"]
        ingest["<b>ingest</b><br/>從佇列認領文件<br/>解析 → 產生向量 → 寫入 Milvus<br/>LiteParse 在這裡執行"]
        embedder["<b>embedder</b><br/>embeddinggemma-2<br/>文字與圖片的向量<br/>CPU（可改 GPU）"]
        vllm["<b>vllm</b>（選用）<br/>LightOnOCR，GPU"]
        monitor["<b>monitor</b><br/>每 5 秒取樣<br/>容器與 GPU 用量"]
        subgraph vdb["向量資料庫"]
            milvus["<b>milvus</b><br/>向量查詢＋BM25"]
            etcd["etcd<br/>設定"]
            minio["minio<br/>資料檔"]
        end
        subgraph storage["共用的儲存"]
            appdb[("app.db（SQLite，WAL）<br/>帳號、對話、知識庫、文件佇列<br/>解析器狀態")]
            mondb[("monitor.db（SQLite）<br/>事件與資源取樣")]
            files[("volumes/kb<br/>原始檔、頁面圖、裁切的圖片")]
        end
    end

    subgraph outside["Docker 外"]
        gemini["Gemini API（雲端）"]
        ollama["Ollama（主機上）"]
        dockersock["Docker socket<br/>nvidia-smi"]
    end

    user -- "HTTP :8000" --> web
    web -- "聊天（串流）" --> gemini
    web -- "聊天（串流）" --> ollama
    web -- "問題轉向量" --> embedder
    web -- "向量＋關鍵字查詢" --> milvus
    web -- "讀寫" --> appdb
    web -- "讀取附圖" --> files
    web -- "寫入對話事件、讀取監控資料" --> mondb

    ingest -- "每 2 秒查佇列、更新狀態<br/>發佈解析器狀態" --> appdb
    ingest -- "讀原始檔、寫頁面圖" --> files
    ingest -- "整頁圖逐頁辨識" --> vllm
    ingest -- "chunk 轉向量" --> embedder
    ingest -- "寫入 chunk" --> milvus
    ingest -- "寫入文件處理事件" --> mondb

    milvus --> etcd
    milvus --> minio

    monitor -- "讀取用量" --> dockersock
    monitor -- "寫入取樣" --> mondb
```

重點：

- **web 和 ingest 不直接互相呼叫**。web 把文件放進 SQLite 的佇列（狀態「排隊中」），ingest 每 2 秒去查一次。解析器能不能用，也是 ingest 寫進 SQLite、web 去讀。
- **只有 web 對外**（`:8000`）。embedder、vllm、milvus 只綁在本機 `127.0.0.1`，其他服務完全不對外。
- **兩個 SQLite 檔和 `volumes/kb` 由 web、ingest 共用**。web 與 ingest 掛載同一個 volume；monitor 只用 `monitor.db`。
- **聊天模型在 Docker 外**：預設用 Gemini，也可以在「系統設定」頁改用主機上的 Ollama。
- **vllm 是選用的**：只有加上 `--profile lightonocr` 時才啟動，用 LightOnOCR 解析時才需要。

## 時序圖：上傳文件到知識庫

以一份 PDF 為例。txt／md 不經過解析器，直接切段。

```mermaid
sequenceDiagram
    autonumber
    actor U as 使用者（瀏覽器）
    participant W as web
    participant DB as app.db（SQLite）
    participant F as volumes/kb
    participant I as ingest
    participant V as vllm（LightOnOCR）
    participant E as embedder
    participant M as milvus

    U->>W: POST /api/kbs/{id}/documents（上傳檔案）
    W->>W: 檢查副檔名、大小
    W->>DB: 讀解析器狀態（ingest 發佈的）
    Note over W: 解析器不能用就直接回 422
    W->>F: 寫入暫存檔，確認讀得開
    W->>DB: 建立文件紀錄（pending）
    W->>F: 暫存檔改成正式檔名
    W->>DB: 狀態改成「排隊中」（queued）
    W-->>U: 回傳文件資料（排隊中）

    loop 每 2 秒
        U->>W: GET 文件清單（看狀態）
        W->>DB: 讀文件狀態
        W-->>U: 目前狀態
    end

    loop 每 2 秒，或上一份處理完時
        I->>DB: 認領最早排隊的文件
    end
    DB-->>I: 文件 id，狀態改成「解析中」（parsing）
    Note over I: 處理期間另一個 thread 定時更新心跳；<br/>ingest 當掉的話，逾時後文件會重新排隊

    I->>F: 讀原始檔
    alt LiteParse
        I->>I: pdfium 轉 400 DPI 頁面圖、抽文字、Tesseract OCR
        I->>F: 寫入整頁圖、內嵌圖片
    else LightOnOCR
        par 同時送最多 4 頁（OCR_CONCURRENCY）
            I->>V: 一頁的圖片
            V-->>I: markdown 文字與圖片區塊座標
        end
        I->>F: 寫入裁切出來的圖片
    end

    I->>DB: 狀態改成「建立索引中」（indexing）
    loop 分批
        I->>E: POST /embed（文字 chunk；整頁圖＋文字每批 2 張）
        E-->>I: 向量
    end
    Note over I,E: embedder 暫時連不上時，等 10、30、60 秒各重試一次；<br/>embedder 一次運算超過 3 分鐘會自行結束、由 Docker 重啟
    I->>M: 寫入 chunk（文字、向量、圖片清單）
    Note over M: Milvus 自動從文字建立 BM25 索引
    I->>DB: 狀態改成「完成」（ready）和 chunk 數
    Note over I,DB: 任一步失敗則改成「失敗」並記下原因，可在知識庫頁重試

    U->>W: GET 文件清單
    W-->>U: 完成
```

## 時序圖：對話

以「選了知識庫的對話」為例。沒選知識庫時，跳過檢索那一段。

```mermaid
sequenceDiagram
    autonumber
    actor U as 使用者（瀏覽器）
    participant W as web
    participant DB as app.db（SQLite）
    participant E as embedder
    participant M as milvus
    participant F as volumes/kb
    participant L as 聊天模型（Gemini／Ollama）

    U->>W: POST /api/conversations/{id}/messages
    W->>DB: 取得「正在回覆」的鎖
    Note over W,DB: 同一個對話已在回覆中就回 409；<br/>鎖放在 SQLite，2 個 worker 都看得到
    W->>DB: 存使用者訊息、讀對話歷史、讀模型設定

    opt 對話有選知識庫
        W->>E: 問題轉向量
        E-->>W: 查詢向量
        W->>M: 向量查詢
        M-->>W: 前 5 筆
        W->>M: BM25 關鍵字查詢
        M-->>W: 前 5 筆
        W->>W: 兩邊分數正規化後加權相加（KB_VECTOR_WEIGHT），取前 5 筆
        W->>F: 讀取參考資料的附圖（最多 4 張）
    end

    W->>L: 系統提示詞＋參考資料＋對話歷史＋附圖（串流）
    Note over W,L: 等待回應時，這個 worker 會先去處理其他請求（async）
    L-->>W: 第一段文字
    Note over W: 第一段就出錯的話回 503，例如配額用完
    W-->>U: 開始串流：先送參考資料（sources）
    loop 直到回答結束
        L-->>W: 下一段文字（思考或回答）
        W-->>U: 轉送這一段
    end
    W->>DB: 存模型的回答
    W-->>U: done（訊息 id）
    W->>DB: 釋放鎖
    Note over W: 各階段耗時寫進 monitor.db，顯示在監控頁
```

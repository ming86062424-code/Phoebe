# embedder 的 CPU／GPU 兩種模式

日期：2026-10-24

> **之後的變更**（2026-10-27 補註）：以下內容是當時的設計，保留作為紀錄。後來改變的部分：
>
> - 已實作完成。ingest 送整頁圖給 embedder 的批次大小在 2026-10-27 統一為一次 2 張，CPU 與 GPU 模式相同（`docs/gpu-memory.md`）。

## 目標

RTX 2060（6 GB）放不下 vLLM 和 GPU 上的 embedder，所以依情況切換：

- **用 LightOnOCR 時**：vLLM 用 GPU，embedder 用 CPU（現狀，預設）。
- **不用 LightOnOCR 時**：不開 vLLM，embedder 用 GPU。LiteParse 建立索引時，整頁圖的向量在 CPU 上約要 100 秒（12 頁），改用 GPU 預期只要幾秒，提問時的查詢向量也會更快。

## 決定

1. **裝置設定**：embedder 新增 `EMBED_DEVICE`，值為 `cpu`（預設）或 `cuda`。
   - 兩種模式都用 float32，因為模型卡禁止 float16，而 RTX 2060 不支援 bfloat16。
   - 指定 `cuda` 但容器看不到 GPU 時，模型載入失敗，記錄原因並結束行程，讓 Docker 重啟。沿用現有的載入失敗流程，不自動改用 CPU，避免使用者以為在用 GPU。
   - 不合法的值在啟動時就拒絕，和 `EMBED_IMAGE_TOKENS` 的處理方式相同。
   - `/health` 回報目前用的裝置。
2. **兩種映像**：同一個 Dockerfile，用 build 參數 `TORCH_VARIANT=cpu|cu126` 選擇 PyTorch 版本。
   - CPU 版：`phoebe-embedder:local`，不變，約 2 GB。
   - GPU 版：`phoebe-embedder:gpu`，CUDA 12.6 版的 PyTorch（實際 `docker images` 顯示 3.93 GB）。只有使用 GPU 模式的人才需要建置。
   - 依賴管理：`embedder/pyproject.toml` 用兩個互斥的 dependency group（`cpu`、`cu126`），各自從 PyTorch 的 CPU 或 cu126 套件來源安裝。`cpu` 是預設 group，本機跑測試時仍裝 CPU 版。
   - 選 cu126 而不是 cu130：兩者都有 torch 2.14 的 wheel，cu126 支援的顯卡和驅動範圍比較廣。
3. **GPU 專用的 compose 設定檔** `docker/docker-compose.gpu.yml`：只覆寫 embedder，包括 build 參數、映像名稱、`EMBED_DEVICE=cuda` 和 GPU 需求。
   - GPU 模式：`docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`
   - LightOnOCR 模式：`docker compose --profile lightonocr up -d`（不帶 GPU 設定檔）
   - 兩者不能同時使用，文件要寫清楚切換步驟。compose 沒辦法擋住同時使用的情況，同時用時 vLLM 或 embedder 會因顯存不足而啟動失敗。
4. **向量不變**：CPU 和 GPU 都是 float32，向量只有極小的數值誤差，**不需要重建索引**。實作後要實測兩種裝置的 cosine 相似度。
5. 不改 web、ingest：它們只透過 HTTP 呼叫 embedder，不知道 embedder 用的是哪種裝置。

## 不做

- 不做自動偵測後切換（`auto`），理由同上。
- 不做 vLLM 開著時自動把 embedder 改回 CPU。
- 不做 bfloat16 或 int8 等降低精度的版本（2026-10-24 已決定）。

## 驗證

- 單元測試：`EMBED_DEVICE` 的驗證、載入時傳入的裝置、compose 設定檔與 Dockerfile 的內容。
- 實測：
  - 建置 GPU 映像，用 GPU 模式啟動，確認 `/health` 回報 `cuda`。
  - 量測 VRAM 和主機記憶體的用量。
  - 比較 CPU 和 GPU 的向量。
  - 用草莓園 PDF 跑 LiteParse，比較建立索引的時間。
  - 切回 LightOnOCR 模式後確認一切正常。
- 文件：`docs/gpu-memory.md` 補上 embedder 的部分和切換方式；`docker/README.md`、`docs/services.md` 同步更新。

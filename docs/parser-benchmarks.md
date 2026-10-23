# 解析器測試紀錄

在這台開發機上測試知識庫解析器的紀錄，之後調整設定或換硬體時可以拿來對照。vLLM 本身的設定與排錯過程見 [vllm-rtx2060.md](vllm-rtx2060.md)，各服務的角色見 [services.md](services.md)，解析器的運作方式見 [parsers.md](parsers.md)。

## 測試環境

| 項目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 2060，6 GB（vLLM 用約 5.2 GB） |
| 主機記憶體 | 32 GB |
| Docker 虛擬機記憶體 | **15.5 GB**（WSL2 預設只用主機的一半，未設定 `.wslconfig`） |
| 同時在跑的其他程式 | RAGFlow（另一個專案，約 3–5 GB 記憶體） |
| 測試檔案 | `test_data/草莓園入園方案.pdf`，12 頁，以圖片為主的宣傳資料 |

## 名詞：一份文件的兩個階段

文件狀態依序是「解析中」→「建立索引中」→「完成」：

- **解析**：解析器把 PDF 變成文字和圖片，每頁一個 chunk。
- **建立索引**：把每個 chunk 交給 embedder 轉成向量，再連同文字寫入 Milvus（Milvus 會從文字自動產生 BM25 關鍵字索引）。時間幾乎都花在算向量上。

## LiteParse 與 LightOnOCR 比較（2026-10-23）

| | 解析 | 建立索引 | 總時間 |
|---|---|---|---|
| **LiteParse** | 75 秒（每頁約 6 秒） | 約 100 秒 | **約 176 秒** |
| **LightOnOCR**（一次送 4 頁） | 約 280–290 秒（每頁約 24 秒） | 約 3 秒 | **約 279 秒** |

- **LiteParse**：解析很快（CPU 上的 pdfium 與 Tesseract），但建立索引慢。它每頁的向量是由「**整頁圖＋文字**」產生的，embedder 在 CPU 上處理圖片很耗時。
- **LightOnOCR**：時間幾乎都花在 GPU 辨識。向量只用文字，所以建立索引只要幾秒。
- 資料來源：
  - LiteParse 的解析時間取自 10-23 單獨計時的那次。
  - LiteParse 的總時間 176 秒取自 10-22 關掉 vLLM 時成功的那次。
  - LightOnOCR 取自 10-23 的速度實測（`OCR_CONCURRENCY=4`）。
  - 10-23 想在同樣的條件下兩個都完整跑一次，但兩份都在建立索引時因 embedder 被終止而失敗（見下方），所以表中的數字是不同次測試拼起來的。

## LightOnOCR 一次送幾頁（`OCR_CONCURRENCY`，2026-10-23）

| 一次送幾頁 | 總時間 | 每頁平均 |
|---|---|---|
| 1 | 630 秒 | 52.5 秒 |
| 2 | 365 秒 | 30.4 秒 |
| 4 | 279 秒 | 23.3 秒 |

預設因此設為 4。詳細的記憶體數據見 [vllm-rtx2060.md](vllm-rtx2060.md#一次送多頁的速度實測2026-10-23)。

## 同時處理多份文件（`INGEST_CONCURRENCY`，2026-10-23）

- **設為 2，4 份 PDF，用 LiteParse，vLLM 關閉時**：最多同時處理 2 份，全部完成，總共 376 秒。記憶體高峰：ingest 3.2 GB、embedder 5.0 GB。
- **設為 2，vLLM 和 RAGFlow 都開著時**：虛擬機記憶體不足，embedder 被終止，2 份文件失敗。
- LiteParse 同時處理兩份時，ingest 的記憶體會從約 0.3 GB 漲到 3.2 GB（兩份都在轉 400 DPI 的頁面圖、跑 Tesseract）。

## 記憶體不足造成的失敗紀錄

都是**主機記憶體**（Docker 虛擬機的 15.5 GB）不夠，不是顯存。embedder 在 CPU 上執行，用的是主機記憶體；它被系統終止後會自動重啟，但當下正在建立索引的文件會失敗（「無法連線到 embedding 服務」）。

| 日期 | 情境 | 結果 |
|---|---|---|
| 10-21 | 第一次上傳 PDF（LiteParse），同時 vLLM 正在啟動、RAGFlow 開著 | embedder 被終止，重試後成功 |
| 10-23 | `INGEST_CONCURRENCY=2`，vLLM 和 RAGFlow 開著 | embedder 被終止，2 份文件失敗 |
| 10-23 | `INGEST_CONCURRENCY=1`，依序處理 LiteParse 和 LightOnOCR，vLLM 和 RAGFlow 開著 | embedder 被終止兩次，兩份都在建立索引時失敗 |

待機時的大約用量：vLLM 4.4 GB、embedder 4 GB（算整頁圖向量時會到 5 GB）、RAGFlow 3–5 GB、其他服務約 1.5 GB。

2026-10-24 起 embedder 不再載入音訊編碼器（304M 參數），同樣的測試中常駐記憶體從 3.67 GB 降到 2.48 GB、高峰從 4.82 GB 降到 3.03 GB，向量完全相同。

## 調整時可以參考的方向

- **提高 Docker 虛擬機的記憶體上限**：主機有 32 GB，可以在 `%USERPROFILE%\.wslconfig` 加上下面的設定，然後執行 `wsl --shutdown`，再重新啟動 Docker Desktop：

  ```
  [wsl2]
  memory=24GB
  ```

  這是最直接的解法。
- **降低 embedder 的記憶體高峰**：每批送給 embedder 的整頁圖目前是 8 張（`app/kb/index.py` 的 `IMAGE_BATCH_SIZE`）。改小可以降低高峰，代價是 LiteParse 建立索引稍微變慢。
- **測試時關掉 RAGFlow**，數據會比較乾淨。
- **embedder 暫時連不上時的處理**：2026-10-24 起，建立索引時會等 10、30、60 秒各重試一次，embedder 被終止後自動重啟、載入完模型，文件就能接著完成；重試 3 次都失敗才標為失敗，可以手動按「重試」。

## 怎麼記錄硬體用量

測試時另開一個終端機，在專案根目錄執行：

```
uv run python -m scripts.monitor_resources --only phoebe
```

- 每 2 秒記錄一次各容器的記憶體、CPU，以及 GPU 的顯存和使用率，寫到 `data/usage-<時間>.csv`（`data/` 已被 git 忽略）。
- 測完按 Ctrl-C 停止，會列出每個容器和 GPU 的記憶體高峰，以及發生的時間。也可以用 `--duration 600` 讓它記錄 600 秒後自動停止。
- `--only phoebe` 只記錄名稱包含 `phoebe` 的容器。「全部容器」那一列的總量仍然包含其他專案的容器（例如 RAGFlow），因為它們共用同一台 Docker 虛擬機的記憶體。
- `--interval 1` 可以改成每 1 秒記錄一次。`docker stats` 本身每次約要 1–2 秒，間隔設得再小也不會更密。
- CSV 可以用 Excel 開啟，依 `source` 篩選後畫折線圖，對照測試的時間點，就能看出高峰發生在哪個階段。

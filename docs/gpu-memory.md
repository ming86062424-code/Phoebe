# GPU 顯存怎麼分配

這份文件說明 GPU 顯存（VRAM）的分配方式和原因。數據都是在開發機的 NVIDIA GeForce RTX 2060（6 GB）上量的。vLLM 第一次在這張卡上跑起來的排錯過程見 [vllm-rtx2060.md](vllm-rtx2060.md)，處理速度見 [parser-benchmarks.md](parser-benchmarks.md)。

## 這張卡有多少可以用

| 項目 | 顯存 |
|---|---|
| 整張卡 | 6 GB（6,144 MiB） |
| Windows 桌面與其他程式 | 約 1.0–1.2 GB |
| **實際可用** | **約 4.8–5.0 GB** |

會用到 GPU 的服務：

- **vLLM（LightOnOCR）**：只有啟用 `lightonocr` profile 時才會啟動。
- **主機上的 Ollama**：選用的本機聊天模型。顯存不夠時會自動把部分模型放到 CPU 執行，變慢但不會失敗。

## vLLM：顯存分成三塊

vLLM 啟動時會先量測用量，再配置顯存。2026-10-24 的啟動 log 顯示：

| 用途 | 大小 | 能不能調 |
|---|---|---|
| 模型權重＋CUDA 等固定開銷 | 約 2.2 GiB | 不能。LightOnOCR-3 是 0.8B 參數，fp16 權重約 1.7 GiB |
| 處理圖片時的運算高峰 | 約 1.3 GiB | 很難調。頁面圖最長邊 2048 px，圖片編碼器需要這麼多 |
| **KV cache**（保存每個請求已經處理過的 token） | 由設定決定 | **可以調，是唯一能省的部分** |

### KV cache 要多大才夠

- **一頁需要多少 token**：
  - 頁面圖最長邊 2048 px，模型每 32×32 px 算一個 token。一張 A4 頁面約 64 × 45 ≈ 2,900 個 token。
  - 模型的輸出最多 4,096 個 token（`MAX_NEW_TOKENS`）。
  - 所以一頁最多約 7,000 個 token。
- **同時處理幾頁**：`OCR_VLLM_MAX_NUM_SEQS=4`，最多同時 4 頁，約 28,000 個 token。
- **最低要求**：vLLM 要求 KV cache 至少放得下一個最長的請求，也就是 `max-model-len` 的 18,000 個 token。
- **換算**：這個模型 1 GiB 的 KV cache 約可放 7.7 萬個 token。

所以設定為 **512 MiB**，約 **38,900 個 token**。這能同時放下 4 頁，也滿足最低要求，啟動 log 會顯示 `GPU KV cache size: 38,918 tokens`。

### 為什麼用「指定大小」，不用「比例」

vLLM 有兩種設定方式：

| 設定 | 意思 |
|---|---|
| `--gpu-memory-utilization`（比例） | vLLM 可以用到「整張卡 × 比例」。扣掉權重和運算高峰，剩下的全部給 KV cache |
| `--kv-cache-memory-bytes`（大小） | KV cache 直接用指定的大小，不管比例 |

2026-10-24 以前用的是比例 0.8，vLLM 會佔滿 4.8 GiB，其中 KV cache 有 1.28 GiB（約 9.9 萬個 token），比實際需要多了 3 倍。改用指定大小之後：

| | 之前（比例 0.8） | 現在（KV cache 512 MiB） |
|---|---|---|
| KV cache | 1.28 GiB（約 9.9 萬個 token） | 0.5 GiB（約 3.9 萬個 token） |
| 整張卡的用量（含桌面） | 5,488 MiB | 待機 3,682 MiB、OCR 時最高 3,967 MiB |
| 12 頁的草莓園 PDF（一次 4 頁） | 279 秒 | 252 秒（沒有變慢） |

省下的約 1.8 GB 可以留給其他用途，例如 Ollama。

### 比例還是要設

就算指定了 KV cache 的大小，vLLM 啟動時仍會檢查：**剩餘顯存要大於「比例 × 整張卡」**，不夠就拒絕啟動。

- 不指定比例時預設 0.92，也就是要 5.5 GiB。6 GB 顯卡扣掉桌面只剩約 5 GiB，**一定會啟動失敗**（2026-10-24 實測）。
- 所以比例設為 **0.7**（4.2 GiB），只當作啟動檢查。實際用量約 2.8 GiB，低於這個門檻。
- 這個檢查能避免顯存被其他程式佔走時，vLLM 勉強啟動，之後在處理文件時才出錯。

### 調整方式

| 情況 | 調整 |
|---|---|
| 想同時處理更多頁（調高 `OCR_VLLM_MAX_NUM_SEQS` 或 `INGEST_CONCURRENCY` × `OCR_CONCURRENCY`） | 依「每頁約 7,000 個 token」調大 `OCR_VLLM_KV_CACHE_MEMORY`，例如 8 頁設 `1G` |
| vLLM 啟動時出現 `Free memory on device … is less than desired GPU memory utilization` | 剩餘顯存不夠。先關掉佔用顯存的程式，例如 Ollama 的模型；或者把 `OCR_VLLM_GPU_UTIL` 再調低一點 |
| 換了更大的顯卡 | 可以調大 KV cache 和同時頁數，並考慮拿掉 `--enforce-eager`，讓 CUDA graph 加速 |

設定都在 `docker/.env`，說明見 `docker/.env.example`。

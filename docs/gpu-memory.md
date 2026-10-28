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
- **embedder 的 GPU 模式**：不用 LightOnOCR 時才使用，和 vLLM 二選一，見下方。
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

## embedder：CPU 和 GPU 兩種模式

6 GB 顯卡放不下 vLLM 和 GPU 上的 embedder，所以依用途切換：

| 模式 | 什麼時候用 | vLLM | embedder | 指令（在 `docker/` 執行） |
|---|---|---|---|---|
| **LightOnOCR 模式**（預設） | 需要 LightOnOCR | GPU | CPU | `docker compose --profile lightonocr up -d` |
| **GPU embedding 模式** | 只用 LiteParse | 不啟動 | GPU | 先 `docker compose --profile lightonocr stop vllm`，再 `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build` |

- **切回 LightOnOCR 模式**：直接執行 `docker compose --profile lightonocr up -d`。embedder 會用 CPU 版的設定重建。
- **不要同時帶 GPU 設定檔和 `--profile lightonocr`**：兩者加起來放不進 6 GB，vLLM 會在啟動檢查時失敗，或者 embedder 在處理時顯存不足。
- GPU 模式下執行一般的 `docker compose up -d`（沒帶 GPU 設定檔），embedder 也會回到 CPU。想讓 GPU 模式一直生效，可以在 `docker/.env` 加上 `COMPOSE_FILE=docker-compose.yml;docker-compose.gpu.yml`（Windows 用分號，Linux／macOS 用冒號）。
- **兩種模式的向量相同**：都是 float32，同一組輸入的 cosine 相似度為 1.0000000，最大差異小於 1e-6（2026-10-24 實測），所以切換不需要重建索引。

### 怎麼做到的

- **兩種映像**：同一個 Dockerfile，用 build 參數 `TORCH_VARIANT` 選 PyTorch 版本。

  | | CPU 版 | GPU 版 |
  |---|---|---|
  | 映像 | `phoebe-embedder:local` | `phoebe-embedder:gpu` |
  | PyTorch | CPU 版 | CUDA 12.6 版（支援 RTX 2060 這類 Turing 顯卡） |
  | `docker images` 顯示的大小 | 2.07 GB | 3.93 GB |

  依賴管理在 `embedder/pyproject.toml`：`cpu`、`cu126` 是兩個互斥的 dependency group，各自從 PyTorch 的 CPU 或 cu126 套件來源安裝。`cpu` 是預設的 group，在本機跑測試時裝的也是 CPU 版。
- **裝置設定** `EMBED_DEVICE`（`cpu` 或 `cuda`）由 compose 決定，不從 `.env` 讀取。設為 `cuda` 但容器看不到 GPU 時，模型載入失敗並重啟，**不會偷偷改用 CPU**，`/health` 會回報目前用的裝置。
- **精度**：兩種模式都用 float32。模型卡禁止 float16，RTX 2060 也不支援 bfloat16。

### 顯存用量（RTX 2060，2026-10-24）

| | 顯存 |
|---|---|
| 模型權重（不含音訊編碼器，float32） | 1.64 GiB |
| 待機（含 CUDA 執行環境） | 約 1.5 GB |
| 處理整頁圖時的高峰 | 約 2.2 GB |

ingest 送整頁圖給 embedder 時**一次只送 2 張**（`app/kb/index.py` 的 `IMAGE_BATCH_SIZE`），embedder 一個請求算一次，每個請求結束後把 PyTorch 快取的顯存還回去。CPU 和 GPU 模式用同一個數字。選 2 的實測（同樣 8 張圖、不同的一次張數）：

| 一次算幾張 | 8 張圖的時間 | 顯存高峰（PyTorch 保留量） |
|---|---|---|
| 8 | 2.73 秒 | 3.12 GiB |
| 4 | 2.70 秒 | 2.39 GiB |
| **2（採用）** | **2.90 秒** | **2.01 GiB** |
| 1 | 3.17 秒 | 1.85 GiB |

- 一開始一次送 8 頁，每頁的文字又長，整張卡的用量最高到 5,888 MiB（含桌面），只剩約 250 MiB。改成一次 2 張後最高約 3,400 MiB，建立索引一樣約 6–7 秒。
- 一次 2 張也降低了 CPU 模式的記憶體高峰：12 頁的草莓園 PDF，embedder 最高從約 3.5 GB 降到約 2.7 GB，建立索引的時間沒有變慢（106–115 秒 → 98–103 秒，2026-10-27）。
- 2026-10-24 時曾改成「ingest 一次送 8 張、embedder 在 GPU 上每次算 2 張」，兩個數字容易混淆，2026-10-27 統一成 ingest 一次送 2 張。
- 請求結束後釋放快取，閒置時顯存會降回約 2.6 GB（含桌面），讓 Ollama 等其他程式可以使用。

### 效果（12 頁的草莓園 PDF，LiteParse）

| | 解析 | 建立索引 | 總時間 |
|---|---|---|---|
| embedder 在 CPU | 約 75 秒 | 約 100 秒 | 約 176 秒 |
| **embedder 在 GPU** | 約 57–64 秒 | **7 秒** | **64–71 秒** |

主機記憶體也從約 2.3 GB 降到約 1.2–1.8 GB，因為模型改放在顯存。

### 和 Ollama 共用

GPU 模式下，embedder 閒置約佔 1.5 GB，處理時約 2.2 GB。主機上的 Ollama 聊天模型（例如 `gemma4:e4b`，6.6 GB）本來就放不進 6 GB 顯卡，會自動把部分模型放到 CPU 執行，embedder 開著時放到 CPU 的比例會更多。聊天建議使用 Gemini。

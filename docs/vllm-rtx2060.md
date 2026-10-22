# 在 RTX 2060（6 GB）上執行 LightOnOCR（vLLM）

這份文件記錄在這台開發機上，讓 LightOnOCR 的 vLLM 服務跑起來時遇到的問題，以及最後採用的設定。之後換顯卡、升級 vLLM 或重灌環境時可以拿來對照。

## 環境

| 項目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 2060，6 GB，Turing 架構 |
| 作業系統 | Windows 11，Docker Desktop（WSL2 後端） |
| vLLM 映像 | `vllm/vllm-openai:v0.30.0`（CUDA 13） |
| 模型 | `lightonai/LightOnOCR-3-0.8B` |
| NVIDIA 驅動 | 580.88（CUDA 13.0） |

## 最後的設定

`docker/docker-compose.yml` 的 `vllm` 服務：

```yaml
command:
  - --model
  - ${OCR_MODEL:-lightonai/LightOnOCR-3-0.8B}
  - --trust-remote-code
  - --limit-mm-per-prompt
  - '{"image": 1}'
  - --default-chat-template-kwargs
  - '{"enable_thinking": false}'
  - --mm-processor-cache-gb
  - "0"
  - --no-enable-prefix-caching
  - --dtype
  - float16                                  # Turing 不支援 bfloat16
  - --gpu-memory-utilization
  - ${OCR_VLLM_GPU_UTIL:-0.8}                # 0.85 會失敗
  - --max-model-len
  - ${OCR_VLLM_MAX_MODEL_LEN:-18000}
  - --enforce-eager                          # 不用 CUDA graph，把顯存留給 KV cache
  - --max-num-seqs
  - ${OCR_VLLM_MAX_NUM_SEQS:-4}              # 預設 256 條序列配不到 KV cache
```

前五個參數（`--trust-remote-code` 到 `--no-enable-prefix-caching`）是 LightOnOCR 官方的 0.8B 設定。`--dtype` 以下都是為了 6 GB 的 Turing 顯卡而調整的。

`docker/.env` 對應的三個值：

```
OCR_VLLM_GPU_UTIL=0.8
OCR_VLLM_MAX_MODEL_LEN=18000
OCR_VLLM_MAX_NUM_SEQS=4
```

用這組設定，GPU 總用量約 4.4 / 6 GiB。KV cache 有 1.28 GiB，可容納 99,243 個 token；以 18000 的長度計算，大約可以同時處理 5.5 條序列。

## 遇到的問題，依發生順序

### 1. 驅動太舊，容器起不來

- **現象**：vllm 容器啟動就失敗，錯誤是 `CUDA driver version is insufficient`（`cudaErrorInsufficientDriver`）。
- **原因**：`vllm/vllm-openai:v0.30.0` 是 CUDA 13 的映像，當時的驅動是 560.94，只支援到 CUDA 12.6。
- **解法**：把 NVIDIA 驅動升級到 580 以上，實測 580.88（CUDA 13.0）。升級後用 `nvidia-smi` 確認右上角的 `CUDA Version` 是 13.0。

### 2. 顯存使用率 0.85：可用顯存不足

- **現象**：

  ```
  ValueError: Free memory on device cuda:0 (4.98/6.0 GiB) on startup is less than
  desired GPU memory utilization (0.85, 5.1 GiB).
  ```

- **原因**：`--gpu-memory-utilization` 是「整張卡」的比例。Windows 桌面本身約佔 1 GiB，所以 vLLM 啟動時只剩 4.98 GiB 可用，不夠 0.85 要求的 5.1 GiB。
- **解法**：`OCR_VLLM_GPU_UTIL` 降到 `0.8`（4.8 GiB）。

### 3. 顯存使用率 0.8：KV cache 配不到空間

- **現象**：

  ```
  ValueError: No available memory for the cache blocks. Try increasing
  `gpu_memory_utilization` when initializing the engine
  ```

- **原因**：
  - vLLM 預設會擷取 CUDA graph，這需要額外的顯存。
  - 預設最多同時處理 256 條序列，每條長度 18000，KV cache 需求很大。
  - 0.8 的額度在模型權重和 CUDA graph 用掉之後，就沒剩空間給 KV cache 了。
  - 錯誤訊息建議調高使用率，但第 2 步已經證明 0.85 不行。
- **解法**：
  - 加上 `--enforce-eager`，不使用 CUDA graph，把那塊顯存留給 KV cache。代價是吞吐量比較低。LightOnOCR 預設一次送 4 頁（`OCR_CONCURRENCY=4`），這個代價就有影響：同時處理的頁越多，每頁越慢（數字見下方〈一次送多頁的速度實測〉）。
  - 加上 `--max-num-seqs`，把同時序列數從 256 降到 8，之後再降為 4（配合「一次送多頁」，見下方的速度實測）。降為 4 後重新量測，KV cache 仍是 1.28 GiB、99,243 tokens。
  - 兩者一起調整後，服務就能正常啟動。

### 4. 不支援 bfloat16

- **原因**：RTX 2060 是 Turing 架構（compute capability 7.5）。bfloat16 要 Ampere（8.0）以上才有硬體支援。
- **解法**：固定 `--dtype float16`。

## 一次送多頁的速度實測（2026-10-23）

LightOnOCR 一份文件可以同時送好幾頁給 vLLM（`OCR_CONCURRENCY`），讓 vLLM 用批次處理（continuous batching）同時算。

測試條件：
- `test_data/草莓園入園方案.pdf`，12 頁，解析器 LightOnOCR
- `--max-num-seqs 4`，`INGEST_CONCURRENCY=1`
- 同一台機器上另有 RAGFlow（約 5 GB 記憶體）在執行

| `OCR_CONCURRENCY` | 總時間 | 每頁平均 | 加速 | 記憶體高峰（ingest／vLLM／embedder） |
|---|---|---|---|---|
| 1（舊的做法，一次一頁） | 630 秒 | 52.5 秒 | 1× | 223 MB／4.8 GB／4.1 GB |
| 2 | 365 秒 | 30.4 秒 | 1.73× | 342 MB／4.6 GB／4.0 GB |
| **4** | **279 秒** | **23.3 秒** | **2.26×** | 377 MB／4.7 GB／4.0 GB |

- 一次 4 頁最快，記憶體只在 ingest 多用約 150 MB，因此 **`OCR_CONCURRENCY` 預設為 4**。
- 加速不是線性的：RTX 2060 的運算量有限，又關掉了 CUDA graph（`--enforce-eager`），同時處理的頁越多，每頁越慢，但總時間仍明顯縮短。
- `INGEST_CONCURRENCY` × `OCR_CONCURRENCY` 是 vLLM 同時收到的請求數。目前 1 × 4 剛好等於 `--max-num-seqs` 的 4；若把 `INGEST_CONCURRENCY` 調高，超過的請求會在 vLLM 內排隊，總處理量不會再增加，只是讓多份文件輪流前進。

## 和主機 Ollama 共用 GPU

- vLLM 會依 `OCR_VLLM_GPU_UTIL` 一次佔走約 4.8 GiB 顯存，而且服務執行期間一直保留。這時主機上的 Ollama 幾乎拿不到顯存。
- 實測 vllm 執行時，用 Ollama 載入 `gemma4:e4b` 聊天會出現 503，錯誤包含 `llama-server reported out-of-memory during startup`。
- 就算載得進去，模型也幾乎都在 CPU 上執行，`/api/ps` 顯示只用到 0.3 GB 顯存，回答很慢。

因此這台的配置是 **GPU 專給 OCR，聊天用雲端 Gemini**（`LLM_PROVIDER=gemini`）。如果要用本機 Ollama 聊天，有兩種做法：

- 處理完文件後停掉 vllm：`docker compose --profile lightonocr stop vllm`。之後需要 OCR 時再執行 `docker compose --profile lightonocr start vllm`。
- 只使用 LiteParse 解析，不需要 GPU，vllm 不必啟動（它預設就不會啟動）。

## 換成更大顯存的 GPU 時

- 可以調高 `OCR_VLLM_GPU_UTIL` 和 `OCR_VLLM_MAX_NUM_SEQS`。
- 可以拿掉 compose 裡的 `--enforce-eager`，啟用 CUDA graph，吞吐量較高。
- Ampere（RTX 30 系列）以上的顯卡支援 bfloat16，可以把 `--dtype` 改回官方建議的 `bfloat16`。
- 改完後執行 `docker compose --profile lightonocr up -d vllm`，再用 `docker compose logs -f vllm` 確認有出現 KV cache 大小的訊息，沒有出現上面那些錯誤。

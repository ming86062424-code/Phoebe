# Docker 的資料存放方式

這份文件說明 Docker 部署時，每一種資料存在哪裡、為什麼這樣放，以及想改變時怎麼做。設定在 [docker/docker-compose.yml](../docker/docker-compose.yml)，各服務的角色見 [services.md](services.md)。

## 總覽

| 資料 | 存放方式 | 名稱或路徑 | 原因 |
|---|---|---|---|
| SQLite（帳號、對話、知識庫清單、設定的 `app.db`；監控資料 `monitor.db`） | Docker 管理的 volume | `phoebe_phoebe-db` | 資料庫：需要可靠的檔案鎖 |
| Milvus 向量資料 | Docker 管理的 volume | `phoebe_milvus-data` | 資料庫：大量隨機讀寫 |
| etcd（Milvus 的設定儲存） | Docker 管理的 volume | `phoebe_etcd-data` | 資料庫：大量小檔案、需要 fsync |
| minio（Milvus 的資料檔） | Docker 管理的 volume | `phoebe_minio-data` | 資料庫：大量小檔案 |
| embedder 的模型快取 | Docker 管理的 volume | `phoebe_embedder-hf-cache` | 快取：啟動時讀取 1.5 GB 的模型，要讀得快 |
| vLLM 的模型快取 | Docker 管理的 volume | `phoebe_vllm-hf-cache` | 快取：同上，模型約 1.6 GB |
| vLLM 的 Triton 編譯快取 | Docker 管理的 volume | `phoebe_vllm-triton-cache` | 快取：大量小檔案，沒有人需要查看 |
| **上傳的檔案、整頁圖、裁切圖** | **主機目錄** | `docker/volumes/kb/` | 使用者會想直接打開查看；檔案以讀取為主，速度影響不大 |

原則：

- **資料庫和快取**放在 Docker 管理的 volume。
- **使用者會想直接查看的檔案**放在主機目錄 `docker/volumes/`（已被 git 忽略）。

Docker 管理的 volume 在 compose 裡的名稱前面會加上專案名稱 `phoebe_`，例如 compose 裡寫的 `milvus-data`，實際名稱是 `phoebe_milvus-data`。

## 兩種存放方式的差別

| | Docker 管理的 volume | 主機目錄（bind mount） |
|---|---|---|
| compose 寫法 | `milvus-data:/var/lib/milvus` | `./volumes/milvus:/var/lib/milvus` |
| 檔案實際在哪 | Docker 自己管理的位置 | 你指定的資料夾 |
| 從檔案總管查看 | 不行，要透過 `docker` 指令 | 可以 |
| 備份 | 用 `docker run … tar` 打包 | 直接複製資料夾 |
| `docker compose down -v` | **會被刪除** | 不受影響 |

## 為什麼在 Windows 上要區分

在 Windows 上，Docker Desktop 的容器其實跑在一台 **WSL2 的 Linux 虛擬機**裡，兩種存放方式的路徑完全不同：

```
Docker 管理的 volume：容器 → Linux 虛擬機 → ext4（虛擬機自己的虛擬硬碟）
主機目錄：            容器 → Linux 虛擬機 →【9P 檔案共享，跨過虛擬機邊界】→ Windows NTFS
```

主機目錄的每一次讀寫，都要透過 9P 協定在虛擬機和 Windows 之間來回傳遞，所以會有兩個問題：

1. **慢**：
   - 2026-10-24 的實測：vLLM 從主機目錄讀取 1.59 GB 的模型花了 19 秒，約 85 MB/s。log 裡寫著 `Filesystem type for checkpoints: 9P`。
   - safetensors 用記憶體映射（mmap）讀檔，在 9P 上特別沒效率。
   - 大量小檔案的情況更明顯，例如 etcd、minio、Triton 快取。
2. **檔案鎖不可靠**：SQLite 的 WAL 模式需要共用記憶體和檔案鎖，SQLite 官方不支援在網路檔案系統上使用 WAL。web 和 ingest 兩個容器會同時讀寫同一個 SQLite，放在主機目錄有損壞的風險。

Docker 管理的 volume 是虛擬機內部原生的 Linux 檔案系統，沒有這兩個問題。

上傳的檔案留在主機目錄，原因是：
- 它們以「寫一次、之後讀取」為主，速度差異不明顯。
- 你會想直接打開確認，例如解析出的圖片。

## 換到 Linux 上執行時

如果是在 Linux 上直接安裝 **Docker Engine**（不是 Docker Desktop），就沒有虛擬機和 9P，兩種方式的速度和檔案鎖都一樣可靠。這時候選哪一種，只看管理上的方便：

- 想全部集中在專案資料夾，可以全部改回主機目錄（做法見下方），SQLite 也可以。
- 要注意**檔案擁有者**：容器裡的程式寫進主機目錄的檔案，在主機上會屬於容器內的使用者。例如 embedder、web、ingest 以 UID 1000 執行，vLLM 和 Milvus 以 root 執行。主機目錄要讓對應的使用者有寫入權限。
- RHEL、Fedora 等啟用 SELinux 的系統，主機目錄可能要加上 `:z` 標記。

如果在 Linux 上用的是 **Docker Desktop**，它一樣會開虛擬機，只是檔案共享用的是比 9P 快的 virtiofs，差距比 Windows 小，但仍然存在。

## Triton 編譯快取

vLLM 會用 Triton 在啟動時現場編譯 GPU 程式，結果存在容器裡的 `/root/.triton/cache`，約 177 MB、181 個檔案。

- 如果沒有掛載這個目錄，**每次重建容器都要重新編譯**。2026-10-24 的實測中，啟動的「量測記憶體與編譯」階段花了約 5 分鐘，整個啟動約 8 分鐘。
- `docker compose restart vllm` 只是重啟，容器還在，快取不會消失。但 `down` 之後再 `up`、改了設定或升級映像時，容器都會重建。
- 2026-10-24 改用 volume 之後的實測（RTX 2060，同時改用 Docker 管理的 volume 讀模型）：

  | | 讀取模型權重 | 初始化引擎（量測、編譯、預熱） | 從容器啟動到可以使用 |
  |---|---|---|---|
  | 之前（主機目錄、沒保存 Triton 快取） | 19 秒 | 371–390 秒 | 約 8 分鐘 |
  | 第一次啟動（volume 是空的，要編譯一次） | 1.5 秒 | 390 秒 | 約 8 分鐘 |
  | 之後重建容器（沿用 Triton 快取） | 1.8 秒 | **30 秒** | **約 1 分鐘** |

- vLLM 官方文件沒有特別要求掛載這個目錄。不過 Triton 支援用 `TRITON_CACHE_DIR` 指定位置，社群常見的做法也是把它掛載出來保存。我們直接掛載預設位置 `/root/.triton/cache`。
- **升級 vLLM 映像後**，舊的快取用不到也不會出錯，Triton 會依程式內容自動判斷要不要重新編譯。想清空的話，刪掉這個 volume 就好（見下方）。

## 常用操作

以下指令都在 `docker/` 目錄執行。在 Git Bash 裡，指令前面要加 `MSYS_NO_PATHCONV=1`，PowerShell 則不需要。

**列出 volume、查看內容**

```
docker volume ls
docker run --rm -v phoebe_milvus-data:/data alpine ls -la /data
```

**刪除快取**（模型會在下次啟動時重新下載，Triton 會重新編譯）

```
docker compose --profile lightonocr down
docker volume rm phoebe_vllm-triton-cache
```

**備份與還原**：見 [docker/README.md](../docker/README.md) 的「資料備份」。

## 想改回主機目錄

compose 裡每個服務原本的主機目錄寫法都還留著，只是被註解掉了：

```yaml
    volumes:
      - milvus-data:/var/lib/milvus
      # - ./volumes/milvus:/var/lib/milvus
```

要改回去時：

1. 把上面那行註解掉，把下面那行的註解拿掉。
2. 先把 volume 裡的資料複製到主機目錄，不然會以空資料啟動。複製方式和下一節的方向相反，把 `/from`、`/to` 對調即可。
3. SQLite（`phoebe-db`）在 Windows 上請不要改到主機目錄，原因見上方的「檔案鎖不可靠」。

## 從舊版（主機目錄）搬到 Docker 管理的 volume

2026-10-24 以前，除了 SQLite 之外的資料都放在 `docker/volumes/` 下。更新到新版之後，Milvus 和模型會改讀 Docker 管理的 volume，**如果沒有搬移資料，知識庫的向量會是空的，模型也會重新下載**。

搬移步驟（在 `docker/` 目錄執行，以 PowerShell 為例；Git Bash 要在每行 `docker run` 前加 `MSYS_NO_PATHCONV=1`）：

```
docker compose --profile lightonocr down
docker compose --profile lightonocr create

docker run --rm -v "${PWD}/volumes/etcd:/from:ro" -v phoebe_etcd-data:/to alpine cp -a /from/. /to/
docker run --rm -v "${PWD}/volumes/minio:/from:ro" -v phoebe_minio-data:/to alpine cp -a /from/. /to/
docker run --rm -v "${PWD}/volumes/milvus:/from:ro" -v phoebe_milvus-data:/to alpine cp -a /from/. /to/
docker run --rm -v "${PWD}/volumes/embedder-hf-cache:/from:ro" -v phoebe_embedder-hf-cache:/to alpine cp -a /from/. /to/
docker run --rm -v "${PWD}/volumes/vllm-hf-cache:/from:ro" -v phoebe_vllm-hf-cache:/to alpine cp -a /from/. /to/
docker run --rm -v phoebe_embedder-hf-cache:/to alpine chown -R 1000:1000 /to

docker compose --profile lightonocr up -d
```

- `create` 只建立容器和 volume，不啟動服務，確保複製時資料庫沒有在寫入。
- 最後一行 `chown` 是因為 embedder 以 UID 1000 執行，需要能寫入自己的模型快取。
- 確認知識庫和對話都正常後，舊的 `docker/volumes/etcd`、`minio`、`milvus`、`embedder-hf-cache`、`vllm-hf-cache` 就可以刪除。`docker/volumes/kb/` **要保留**，它仍然在使用中。
- 不搬也可以：模型會重新下載，向量可以用 `reprocess_all` 從 `docker/volumes/kb/` 裡的原始檔重建，做法見 [docker/README.md](../docker/README.md)。

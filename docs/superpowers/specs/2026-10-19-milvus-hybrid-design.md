# 知識庫改用 Milvus 混合查詢（關鍵字＋向量）— 設計規格

- 日期：2026-10-19
- 子專案：3 / 4（部署基礎 → LLM 供應商 → **Milvus 混合查詢** → 多模態知識庫）
- 建立在子專案 2（`feature/llm-providers`，PR #7）之上
- 狀態：使用者授權直接開發

## 1. 目的與範圍

**目的**：把知識庫的向量庫從 Chroma 換成 Milvus，檢索改為「BM25 關鍵字 ＋ 向量」混合查詢，提升專有名詞、編號、代碼等精確字詞的命中率。

**需求（使用者確認）**
1. 向量資料庫改用 Milvus；快速測試時 Milvus 也以 Docker 執行（Milvus Lite 不支援 Windows）
2. 混合查詢：BM25 關鍵字 ＋ dense 向量，以 **RRF** 合併
3. 中文斷詞使用 Milvus 內建 **jieba**（`chinese` analyzer）
4. 既有 Chroma 資料提供**一次性遷移**（不需重新 OCR），之後完全移除 Chroma
5. embedding 模型維持本機 Ollama（`EMBED_MODEL`，預設 embeddinggemma）

**明確不做（YAGNI）**：引用介面顯示分數與排序（子專案 4，但本子專案的檢索結果要提供所需分數）、圖片與多種檔案格式（子專案 4）、rerank 模型、分散式 Milvus、權重可調的融合。

## 2. 架構

### 2.1 分層
```
app/kb/
├── index.py          # KnowledgeIndex：切塊、embedding、RRF 融合、回傳 Source（公開介面不變）
├── chunk_store.py    # ChunkStore 協定、ChunkRecord、Hit
├── milvus_store.py   # MilvusChunkStore（pymilvus MilvusClient）
└── migrate_chroma.py # python -m app.kb.migrate_chroma：一次性遷移
tests/fakes_store.py  # InMemoryChunkStore（單元測試用，關鍵字以簡單分詞計分）
```

`KnowledgeIndex` 的公開方法維持不變：`add_document(kb_id, document_id, filename, pages) -> int`、`delete_document(kb_id, document_id)`、`delete_kb(kb_id)`、`search(kb_id, query, top_k) -> list[Source]`。worker、聊天、API 不需改呼叫方式。

### 2.2 ChunkStore 協定
- `ChunkRecord(kb_id, document_id, filename, page, chunk_index, text, dense: list[float])`
- `Hit(chunk_id: str, kb_id, document_id, filename, page, text, score: float)`
- 方法：`ensure_ready(dim: int)`、`insert(records)`、`delete_document(kb_id, document_id)`、`delete_kb(kb_id)`、`dense_search(kb_id, vector, limit) -> list[Hit]`、`keyword_search(kb_id, query, limit) -> list[Hit]`

### 2.3 Milvus collection（單一 collection）
- 名稱：`MILVUS_COLLECTION`（預設 `phoebe_chunks`）
- 欄位：
  - `id` VARCHAR(64) 主鍵 = `"{document_id}:{chunk_index}"`
  - `kb_id` INT64，**partition key**
  - `document_id` INT64、`page` INT64、`chunk_index` INT64
  - `filename` VARCHAR(2048)（長度以 UTF-8 位元組計，寫入時以位元組安全方式截斷）
  - `text` VARCHAR(65535)（長度以 UTF-8 位元組計；8192 字元 × 4 位元組可容納，超過仍以位元組安全方式截斷），`enable_analyzer=True`，`analyzer_params` 為 `TEXT_ANALYZER`：jieba 斷詞，filter 依序為 `cnalphanumonly`（去掉標點）、`lowercase`、`stop`（英文疑問詞／助動詞與中文虛詞停用詞表，見 `app/kb/milvus_store.py`），避免自然語言問句的疑問詞主導 BM25
  - `dense` FLOAT_VECTOR(dim)
  - `sparse` SPARSE_FLOAT_VECTOR，由 BM25 Function（輸入 `text`）自動產生
- 索引：`dense` → AUTOINDEX、metric COSINE；`sparse` → SPARSE_INVERTED_INDEX、metric BM25
- 搜尋一律 `filter="kb_id == {kb_id}"`、`consistency_level="Strong"`（小規模、剛索引完就要查得到）
- `ensure_ready(dim)`：collection 不存在就建立；存在但 `dense` 維度不同 → 拋錯「EMBED_MODEL 的向量維度（{dim}）與既有知識庫（{existing}）不同，請重建知識庫」；`text` 欄位的 analyzer 設定與 `TEXT_ANALYZER` 不同 → 拋錯「知識庫的斷詞設定已更新，請重建知識庫（刪除 collection 後重新遷移或重新上傳）」；存在但未載入時呼叫 `load_collection`
- 維度：第一次使用（索引或檢索）時，以實際 embedding 向量的長度檢查，不在啟動時；失敗不快取，下次呼叫會重試

### 2.4 混合查詢與 RRF
- `search`：query 向量化 → `dense_search(limit=top_k*4)` 與 `keyword_search(limit=top_k*4)` → RRF：`score = Σ 1/(60 + rank)`（rank 從 1 起）→ 取前 `top_k`
- `Source` 新增欄位：`vector_score: float | None`（cosine）、`keyword_score: float | None`（BM25）；`score` 為 RRF 分數；`to_dict()` 一併輸出（子專案 4 會用）
- 兩路都沒有結果 → 空清單

### 2.5 寫入
- 切塊：沿用 `SentenceSplitter(chunk_size=512, chunk_overlap=64)`；embedding 文字沿用現有規則（含檔名，與現在一致）
- `add_document` 先 `delete_document` 再插入（重試時不留舊塊）；批次插入（每批 ≤ 256）
- `text` 超過 8192 字元時截斷（記錄警告）

### 2.6 設定
| 變數 | 預設 | 說明 |
|---|---|---|
| `MILVUS_URI` | `http://localhost:19530` | Docker 內為 `http://milvus:19530`（compose 固定） |
| `MILVUS_TOKEN` | （空） | 有啟用驗證時使用 |
| `MILVUS_COLLECTION` | `phoebe_chunks` | |

移除 `CHROMA_DIR`、`CHROMA_HOST`、`CHROMA_PORT`（遷移指令以參數指定來源）。依賴：新增 `pymilvus`（固定發布超過兩週的版本）；移除 `chromadb`、`llama-index-vector-stores-chroma`（`chromadb` 移到 `migrate` dependency group，只供遷移使用）。

### 2.7 Docker
- compose 新增官方 Milvus standalone 所需的三個服務：`milvus-etcd`、`milvus-minio`、`milvus`（版本固定、發布超過兩週、支援 BM25 Function 與 chinese analyzer），named volumes；`milvus` 有 healthcheck（`/healthz` on 9091）
- `web`、`worker` 依賴 `milvus` healthy；環境固定 `MILVUS_URI=http://milvus:19530`
- 移除 `chroma` 服務；`chroma-data` volume 保留於 profile `migrate-chroma` 的 `chroma` 服務，供遷移時讀取
- 快速測試：`cd docker && docker compose up -d milvus`（連帶 etcd、minio），本機 `MILVUS_URI=http://localhost:19530`；milvus 的 19530 只綁 `127.0.0.1`

### 2.8 一次性遷移
- `python -m app.kb.migrate_chroma (--chroma-dir PATH | --chroma-host HOST [--chroma-port 8000]) [--dry-run]`（需 `uv sync --group migrate`）
- 讀取每個 `kb_<id>` collection 的所有 chunk（文字、`kb_document_id`、`filename`、`page`），依 document 分組、依原順序給 `chunk_index`，重新 embedding 後寫入 Milvus（先刪該 document 既有塊，可重複執行）
- 只遷移 SQLite 中仍存在的 kb 與 document；結束時印出每個 kb 的 chunk 數與略過數
- Docker：`cd docker && docker compose --profile migrate-chroma up -d chroma` 後 `docker compose run --rm worker python -m app.kb.migrate_chroma --chroma-host chroma`（README 說明；worker 映像需含 `migrate` group 的 chromadb——或提供單獨 `migrate` target，擇簡）

## 3. 錯誤處理
- Milvus 連不上：知識庫檢索失敗沿用現有 503「知識庫檢索失敗：…」；索引時文件標 failed，錯誤「無法連線到 Milvus」
- 維度不符：第一次使用時偵測；檢索與索引回清楚訊息（§2.3）
- 刪除文件／知識庫時 Milvus 不可用：仍回 204（記錄警告）；檢索結果再以 SQLite 現存文件過濾，已刪除文件不會被引用
- 查詢含特殊字元（引號、反斜線）：keyword_search 以參數傳遞，不拼接 filter 字串；`kb_id` 為整數

## 4. 測試
- 單元（InMemoryChunkStore）：RRF 排名與分數（兩路交集排前、只在一路出現也會出現）、`vector_score`／`keyword_score` 帶入、top_k、空結果、刪除 document／kb、重試不留舊塊、批次插入、超長文字截斷
- MilvusChunkStore：schema 定義（欄位、analyzer、function、索引）以替身 client 驗證呼叫參數；維度不符的錯誤
- 整合（`@pytest.mark.milvus`，`MILVUS_URI` 可連時才跑）：真實 Milvus 建 collection、插入中英文 chunk、`keyword_search("柴犬")` 命中、`dense_search` 命中、混合結果、刪除
- 遷移：以本機 Chroma（PersistentClient 臨時目錄）建立舊格式資料 → 遷移到 InMemoryChunkStore → 數量、欄位、chunk_index 正確；略過已刪除的 document
- 實測：Docker 起 Milvus；上傳中文 PDF（含專有編號，例如「Invoice No. 2026-1018」）→ 以編號搜尋命中（關鍵字路徑）、以語意問句搜尋命中（向量路徑）；聊天引用正常

## 5. 開發流程
- 分支 `feature/milvus`（從 `feature/llm-providers`），PR 目標 `feature/llm-providers`

# 知識庫改用 Milvus 混合查詢 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 知識庫向量庫由 Chroma 換成 Milvus，檢索改為 BM25 關鍵字＋向量、以 RRF 合併；提供 Chroma 一次性遷移。

**Architecture:** `KnowledgeIndex`（公開介面不變）改為依賴 `ChunkStore` 協定；`MilvusChunkStore` 以單一 collection（`kb_id` partition key、`text` 開 chinese analyzer、BM25 Function 產生 sparse、dense COSINE）實作；RRF 在 Python 端融合兩路結果並保留各自分數。

**Tech Stack:** pymilvus（MilvusClient）、Milvus standalone（Docker，etcd＋minio）、llama-index-core SentenceSplitter、Ollama embedding、pytest

**Spec:** `docs/superpowers/specs/2026-10-19-milvus-hybrid-design.md`

## Global Constraints

- `KnowledgeIndex` 公開方法簽章不變：`add_document(kb_id, document_id, filename, pages) -> int`、`delete_document(kb_id, document_id)`、`delete_kb(kb_id)`、`search(kb_id, query, top_k) -> list[Source]`
- 切塊 `chunk_size=512`、`chunk_overlap=64`；embedding 文字規則與現在相同
- RRF：`score = Σ 1/(60 + rank)`，rank 從 1 起；每路取 `top_k*4`
- `Source` 新增 `vector_score: float | None`、`keyword_score: float | None`；`score` 為 RRF；`to_dict()` 含兩者
- Milvus：collection 預設 `phoebe_chunks`；欄位與索引照規格 §2.3；搜尋 `consistency_level="Strong"`、filter 只用整數 `kb_id`
- 錯誤訊息（逐字）：「無法連線到 Milvus」、「EMBED_MODEL 的向量維度（{dim}）與既有知識庫（{existing}）不同，請重建知識庫」
- 新設定：`MILVUS_URI`（`http://localhost:19530`）、`MILVUS_TOKEN`（空）、`MILVUS_COLLECTION`（`phoebe_chunks`）；移除 `CHROMA_DIR`、`CHROMA_HOST`、`CHROMA_PORT`（兩份 `.env.example` 與測試同步）
- 依賴與映像版本固定、發布超過兩週；`chromadb` 只留在 `migrate` dependency group
- 單元測試不得需要真實 Milvus；整合測試標 `@pytest.mark.milvus`，`MILVUS_URI` 連不上就 skip
- Windows Git Bash；uv 在 `/c/Users/anny3/AppData/Roaming/Python/Python312/Scripts`；`PYTHONUTF8=1`；LF 換行；繁體中文訊息與文件
- 每個 commit 結尾：空一行後 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

1. **查詢含引號、反斜線、`or 1==1` 之類字串**：期望當成一般文字搜尋，不會變成 filter 注入或錯誤。→ Task 1 `test_keyword_search_passes_query_as_data_not_filter`
2. **只在關鍵字路徑命中的編號（如 `2026-1018`）**：期望仍出現在結果中。→ Task 2 `test_keyword_only_hit_is_returned`
3. **重新處理同一份文件**：期望不留下舊 chunk（不重複）。→ Task 2 `test_reindex_replaces_old_chunks`
4. **換了 EMBED_MODEL 導致維度不同**：期望清楚錯誤，不是 Milvus 原始錯誤。→ Task 1 `test_dimension_mismatch_raises_clear_error`
5. **遷移重複執行**：期望不會產生重複資料。→ Task 3 `test_migration_is_idempotent`

---

### Task 1: MilvusChunkStore 與設定

**Files:**
- Create: `app/kb/chunk_store.py`、`app/kb/milvus_store.py`、`tests/test_milvus_store.py`
- Modify: `app/config.py`、`.env.example`、`docker/.env.example`、`tests/test_config.py`、`pyproject.toml`、`uv.lock`（`uv add "pymilvus==X"`）、`pyproject.toml` pytest markers（`milvus`）

**Interfaces:**
- Produces:
  - `chunk_store.ChunkRecord`、`chunk_store.Hit`（欄位見規格 §2.2）、`ChunkStore`（Protocol）
  - `MilvusChunkStore(uri: str, token: str, collection: str, client=None)`：`client` 可注入替身（預設 `pymilvus.MilvusClient(uri=..., token=...)`）；實作 `ensure_ready(dim)`、`insert`、`delete_document`、`delete_kb`、`dense_search`、`keyword_search`；連線錯誤 → `KnowledgeStoreError("無法連線到 Milvus")`（定義在 `chunk_store.py`）
  - `Settings.milvus_uri / milvus_token / milvus_collection`；移除 chroma 三個欄位（`app/kb/index.py`、`services.py` 仍引用者在 Task 2 改；本任務若刪欄位造成匯入錯誤，於本任務暫時讓 `services.py` 不傳 chroma 參數——以全套件通過為準）

- [ ] **Step 1: Write the failing tests**（替身 client 記錄呼叫）

```python
def test_ensure_ready_creates_schema_with_bm25_and_chinese_analyzer(...)  # 欄位、analyzer {"type":"chinese"}、BM25 function、兩個索引、partition key
def test_ensure_ready_existing_collection_same_dim_is_noop(...)
def test_dimension_mismatch_raises_clear_error(...)
def test_insert_batches_and_ids(...)                     # id "{doc}:{idx}"、每批 ≤256
def test_delete_document_and_kb_use_integer_filters(...)
def test_dense_search_maps_hits(...)                     # filter "kb_id == 3"、Strong、COSINE 分數
def test_keyword_search_passes_query_as_data_not_filter(...)  # query 含 '"' 與 "\\" 與 "or 1==1" → 出現在 data，不在 filter
def test_connection_error_maps_to_store_error(...)
@pytest.mark.milvus
def test_real_milvus_roundtrip(...)                      # 連不上就 skip；建 collection、插入「柴犬」與英文、keyword 命中、dense 命中、刪除
```

- [ ] **Step 2: Run** `uv run pytest -q tests/test_milvus_store.py` → FAIL
- [ ] **Step 3: Implement**（以已安裝 pymilvus 原始碼確認 `MilvusClient.create_schema`、`Function(function_type=FunctionType.BM25)`、`prepare_index_params`、`search(anns_field=..., search_params={"metric_type": ...})` 等實際名稱）
- [ ] **Step 4: Run** `uv run pytest -q` → 全部通過（milvus 標記測試在無 Milvus 時 skip）
- [ ] **Step 5: Commit** `feat: Milvus chunk store with BM25 and dense fields`

---

### Task 2: KnowledgeIndex 改用 ChunkStore 與 RRF

**Files:**
- Modify: `app/kb/index.py`、`app/kb/services.py`、`app/kb/worker.py`（若有 chroma 相關）、`pyproject.toml`（移除 `chromadb`、`llama-index-vector-stores-chroma` 主依賴）
- Create: `tests/fakes_store.py`
- Test: `tests/test_kb_index.py`（改寫）、`tests/test_kb_api.py`、`tests/test_kb_worker.py`（改用 InMemoryChunkStore）

**Interfaces:**
- Consumes: Task 1
- Produces:
  - `KnowledgeIndex(store: ChunkStore, embed_model: BaseEmbedding)`；建構時以 `embed_model.get_text_embedding("dimension probe")` 取維度並 `store.ensure_ready(dim)`（延遲到第一次使用亦可，但錯誤訊息須為規格文字）
  - `Source` 新欄位；`rrf_fuse(dense: list[Hit], keyword: list[Hit], top_k: int) -> list[tuple[Hit, float, float | None, float | None]]`（純函式，可測）
  - `build_kb_services` 建 `MilvusChunkStore(settings.milvus_uri, settings.milvus_token, settings.milvus_collection)`
  - `tests/fakes_store.InMemoryChunkStore`：dense 以 cosine、keyword 以「query 以空白與 jieba 不可用時的字元二元組」簡單計分即可（只供單元測試）

- [ ] **Step 1: Write the failing tests**

```python
def test_rrf_ranks_hits_in_both_lists_first(...)
def test_keyword_only_hit_is_returned(...)                # 只在 keyword 出現 → 仍在結果，vector_score None
def test_scores_are_carried(...)                          # vector_score / keyword_score / score(RRF) 正確
def test_top_k_and_empty(...)
def test_reindex_replaces_old_chunks(...)                 # add_document 兩次 → chunk 數不變
def test_delete_document_and_kb(...)
def test_long_text_is_truncated_to_8192(...)
def test_source_to_dict_includes_component_scores(...)
```
既有 worker／API／聊天測試改用 `InMemoryChunkStore` 後照常通過。

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement**；`app` 內不得再 import `chromadb`
- [ ] **Step 4: Run** `uv run pytest -q` → 全部通過
- [ ] **Step 5: Commit** `feat: hybrid BM25 + vector retrieval with RRF on Milvus`

---

### Task 3: Chroma 一次性遷移

**Files:**
- Create: `app/kb/migrate_chroma.py`、`tests/test_migrate_chroma.py`
- Modify: `pyproject.toml`（`uv add --group migrate chromadb==<現有鎖定版本>`）

**Interfaces:**
- Consumes: `KnowledgeIndex` / `ChunkStore`、`app.db.connect`、`app.kb.store`
- Produces: `migrate(chroma_client, db_conn, index: KnowledgeIndex, dry_run: bool) -> dict[int, tuple[int, int]]`（kb_id → (遷移 chunk 數, 略過數)）；`main()` 解析參數 `--chroma-dir | --chroma-host [--chroma-port] [--dry-run]`，以 `Settings.from_env()` 建 Milvus 與 embedding

- [ ] **Step 1: Write the failing tests**（臨時目錄 PersistentClient 建舊格式：collection `kb_<id>`、metadata `kb_document_id`/`filename`/`page`）

```python
def test_migrates_chunks_with_order_and_metadata(...)
def test_skips_deleted_documents_and_kbs(...)
def test_migration_is_idempotent(...)                     # 跑兩次 → store 內數量不變
def test_dry_run_writes_nothing(...)
```
（若 `migrate` group 未安裝，測試以 `pytest.importorskip("chromadb")` 跳過）

- [ ] **Step 2: Run** `uv run --group migrate pytest -q tests/test_migrate_chroma.py` → FAIL
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run** `uv run --group migrate pytest -q` → 全部通過；`uv run pytest -q`（無 migrate group）也通過
- [ ] **Step 5: Commit** `feat: one-time Chroma to Milvus migration`

---

### Task 4: Docker、文件與實測

**Files:**
- Modify: `docker/docker-compose.yml`、`docker/Dockerfile`（遷移所需：worker 加 `--group migrate` 或另設 target，擇簡並說明）、`docker/README.md`、`README.md`、`docs/superpowers/specs/2026-10-18-deployment-design.md`（移除 Chroma 敘述的過渡說明）
- Test: `tests/test_config.py`（compose 結構測試：milvus 三服務、healthcheck、web/worker 依賴 milvus healthy、19530 綁 127.0.0.1、無 chroma 預設服務、`migrate-chroma` profile）

- [ ] **Step 1: compose 測試先寫**，Run → FAIL
- [ ] **Step 2: 實作 compose 與文件**（Milvus／etcd／minio 版本固定且發布超過兩週，以 Docker Hub／GitHub release 日期確認並記錄）
- [ ] **Step 3: Run** `uv run pytest -q` → 全部通過
- [ ] **Step 4: Commit** `feat: Milvus in Docker, Chroma kept only for migration`
- [ ] **Step 5: 實測（controller）**：`cd docker && docker compose up -d --build`；整合測試 `MILVUS_URI=http://localhost:19530 uv run pytest -q -m milvus`；上傳含「Invoice No. 2026-1018」的中文 PDF → 以「2026-1018」與語意問句搜尋都命中；聊天引用正常

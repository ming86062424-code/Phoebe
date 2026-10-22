# ingest 改名、並行處理與 vLLM profile — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the background document service from `worker` to `ingest`, let it process several documents and several OCR pages at once (configurable), and put vLLM behind an opt-in `lightonocr` compose profile.

**Architecture:** A dispatcher thread in `Ingester` claims documents while fewer than `INGEST_CONCURRENCY` are in flight and hands each to a `ThreadPoolExecutor`. The LightOnOCR parser renders pages one by one (all pypdfium2 calls under one process-wide lock) and keeps up to `OCR_CONCURRENCY` OCR requests in flight. vLLM gets `profiles: ["lightonocr"]` and `--max-num-seqs 4`.

**Tech Stack:** Python 3.12 (uv), FastAPI, SQLite, pypdfium2, LiteParse 2.16, httpx, Docker Compose 2.29, React/TS (no frontend changes expected).

**Spec:** `docs/superpowers/specs/2026-10-22-ingest-concurrency-design.md`

## Global Constraints

- User-facing messages and docs in Traditional Chinese (zh-TW); code comments and identifiers in English.
- LF line endings; no new runtime dependencies.
- Windows dev box, Git Bash: `export PATH="/c/Users/anny3/AppData/Roaming/Python/Python312/Scripts:$PATH"`, run Python with `PYTHONUTF8=1`.
- Backend checks: `uv run pytest -q` (all green; integration tests may skip without Docker).
- Gunicorn's own "worker" terms (`workers`, `worker_class`, `WEB_CONCURRENCY`, `uvicorn-worker`) are NOT renamed.
- Defaults: `INGEST_CONCURRENCY=1`, `OCR_TIMEOUT_SECONDS=600`, `OCR_VLLM_MAX_NUM_SEQS=4`; `OCR_CONCURRENCY` default decided in Task 6 (1–4).
- Old env vars `KB_WORKER_MODE`, `KB_WORKER_POLL_SECONDS` (non-empty) must make `Settings.from_env()` raise `ValueError` naming the new variable.
- Branch `feature/ingest`; commits end with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; stage explicit paths only (never `.gitignore`, `test_data/`).

## Review Focus

1. A document deleted while another is being processed in parallel: only the deleted one stops; the other finishes normally. → Task 4 test `test_deleting_one_document_leaves_the_parallel_one_alone`.
2. `process()` raising an unexpected exception in a pool thread must still free its slot (no slow leak to zero capacity). → Task 4 test `test_a_crashing_document_frees_its_slot`.
3. Service stop while OCR pages are in flight: no partial chunks are indexed and the document is left for the stale requeue. → Task 5 test `test_stop_while_pages_are_in_flight_indexes_nothing`.
4. Two documents of the same knowledge base indexed at the same time must not delete each other's chunks. → Task 4 test `test_two_documents_of_one_knowledge_base_index_side_by_side`.
5. Parser unavailable while slots are busy: queued documents of that parser still wait instead of failing. → Task 4 test `test_unavailable_parser_still_waits_with_parallel_slots`.

---

### Task 1: Rename worker → ingest (no behavior change)

**Files:**
- Rename: `app/kb/worker.py` → `app/kb/ingest.py`, `app/kb/worker_main.py` → `app/kb/ingest_main.py`, `tests/test_kb_worker.py` → `tests/test_ingest.py` (use `git mv`)
- Modify: `app/kb/services.py`, `app/config.py`, `app/main.py`, `app/routes/kb.py`, `app/kb/store.py`, `app/kb/milvus_store.py`, `app/kb/reprocess_all.py`, `app/deps.py`, `app/chat_stream.py`, `docker/Dockerfile`, `docker/docker-compose.yml`, `.env.example`, `docker/.env.example`, `README.md`, `docker/README.md`, `docs/services.md`, all tests that reference the old names (`tests/test_config.py`, `tests/test_kb_api.py`, …)

**Interfaces:**
- Produces: class `Ingester` in `app/kb/ingest.py` (was `IngestWorker`, same constructor); `KbServices.ingester` (was `.worker`); `Settings.kb_ingest_mode: str = "embedded"`, `Settings.kb_ingest_poll_seconds: float = 2.0` read from `KB_INGEST_MODE` / `KB_INGEST_POLL_SECONDS`; entry point `python -m app.kb.ingest_main`; compose service `ingest`, image `phoebe-ingest:local`, Dockerfile target `ingest`; `WORKER_DOWN_REASON` renamed `INGEST_DOWN_REASON = "文件處理服務（ingest）未啟動"`.

- [ ] **Step 1: Write the failing tests** in `tests/test_config.py`:

```python
@pytest.mark.parametrize(("old", "new"), [
    ("KB_WORKER_MODE", "KB_INGEST_MODE"),
    ("KB_WORKER_POLL_SECONDS", "KB_INGEST_POLL_SECONDS"),
])
def test_old_worker_settings_are_rejected(monkeypatch, old, new):
    monkeypatch.setenv(old, "2")
    with pytest.raises(ValueError, match=f"{old} 已改名為 {new}"):
        Settings.from_env()

def test_ingest_settings(monkeypatch):
    monkeypatch.setenv("KB_INGEST_MODE", "external")
    monkeypatch.setenv("KB_INGEST_POLL_SECONDS", "3")
    s = Settings.from_env()
    assert (s.kb_ingest_mode, s.kb_ingest_poll_seconds) == ("external", 3.0)

def test_compose_runs_the_ingest_service():
    services = _compose()["services"]
    assert "worker" not in services
    assert services["ingest"]["build"]["target"] == "ingest"
    assert services["ingest"]["image"] == "phoebe-ingest:local"
```
Add `KB_WORKER_*` to `RETIRED_KEYS` in `tests/test_config.py`; update every existing compose/Dockerfile assertion from `worker` to `ingest`.

- [ ] **Step 2: Run** `PYTHONUTF8=1 uv run pytest -q tests/test_config.py` → the new tests FAIL.
- [ ] **Step 3: Rename** files with `git mv`, then names, settings, env vars, compose, Dockerfile, messages, log lines (`解析 worker 已啟動` → `ingest 已啟動`), thread names, comments and docs as listed in spec §2. Rewrite comments where "worker" means a Gunicorn process (`app/deps.py`, `app/chat_stream.py` → 「Gunicorn 程序」) or Starlette's threadpool (`app/routes/kb.py:244` → 「背景執行緒」). Add the old-env rejection next to the `ADMIN_USERNAMES` check in `Settings.from_env()`.
- [ ] **Step 4: Verify** `grep -rIn -i "worker" app tests docker README.md docs/services.md .env.example` shows only Gunicorn terms, `uvicorn_worker`, and the retired-name messages; `PYTHONUTF8=1 uv run pytest -q` all pass.
- [ ] **Step 5: Commit** `refactor: rename the worker service to ingest`

---

### Task 2: vLLM behind the `lightonocr` profile, `--max-num-seqs 4`

**Files:**
- Modify: `docker/docker-compose.yml`, `docker/.env.example`, `docker/README.md`, `docs/vllm-rtx2060.md`, `docs/services.md`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: compose service `ingest` (Task 1).

- [ ] **Step 1: Write the failing tests:**

```python
def test_vllm_is_opt_in_behind_the_lightonocr_profile():
    services = _compose()["services"]
    assert services["vllm"]["profiles"] == ["lightonocr"]
    dep = services["ingest"]["depends_on"]["vllm"]
    assert dep == {"condition": "service_started", "required": False}
    command = services["vllm"]["command"]
    assert command[command.index("--max-num-seqs") + 1] == "${OCR_VLLM_MAX_NUM_SEQS:-4}"

def test_docker_env_example_defaults_to_four_sequences():
    text = (PROJECT_ROOT / "docker" / ".env.example").read_text(encoding="utf-8")
    assert "OCR_VLLM_MAX_NUM_SEQS=4" in text
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** the compose changes; `.env.example` value 4; docs: commands `docker compose up -d` / `docker compose --profile lightonocr up -d` / `docker compose --profile lightonocr down` with the warning that `down` without the profile leaves vLLM running; `docs/vllm-rtx2060.md` max-num-seqs 8 → 4 (keep KV-cache numbers, note they will be re-measured in Task 6).
- [ ] **Step 4: Run** `PYTHONUTF8=1 uv run pytest -q tests/test_config.py` → PASS; `cd docker && docker compose config --profiles` lists `lightonocr`; `docker compose config` (no profile) has no `vllm` service.
- [ ] **Step 5: Commit** `feat: start vLLM only with the lightonocr profile`

---

### Task 3: Thread safety for pdfium, LiteParse and the local OCR engine

**Files:**
- Create: `app/kb/parsers/pdfium_lock.py`
- Modify: `app/kb/parsers/lightonocr.py`, `app/kb/parsers/liteparse_parser.py`, `app/kb/parsers/ocr_engines.py`
- Test: `tests/test_parsers.py`

**Interfaces:**
- Produces: `PDFIUM_LOCK: threading.RLock` in `app/kb/parsers/pdfium_lock.py`. Every pypdfium2 call (open, `len`, page access, render, close, `pdf_page_count`) holds it. `TransformersOcrEngine.ocr` holds its own `threading.Lock`. If Step 2's stress test shows LiteParse is unsafe, LiteParse parse and screenshot calls also hold `PDFIUM_LOCK` (decide from evidence, record the result in the commit message).

- [ ] **Step 1: Write tests:**
  - `test_pdfium_calls_hold_the_lock`: monkeypatch `pypdfium2.PdfDocument` with a wrapper asserting `PDFIUM_LOCK._is_owned()` on every call; parse a 3-page `make_pdf` with `LightOnOcrParser(FakeEngine)` and `pdf_page_count`.
  - `test_liteparse_parallel_matches_sequential`: parse 6 copies of a 4-page PDF with `LiteParseParser("eng")` in a `ThreadPoolExecutor(4)` into separate out dirs; each result's `(page, text)` list equals the sequential result and every `page_N.png` exists.
  - `test_transformers_engine_runs_one_ocr_at_a_time`: with a fake model whose `generate` records concurrent entries, 4 threads calling `ocr()` never overlap.
- [ ] **Step 2: Run** → the lock test and the transformers test FAIL; record whether the LiteParse test passes without a lock (run it 5 times).
- [ ] **Step 3: Implement** the lock module and wrap the calls; add the engine lock; add the LiteParse lock only if Step 2 failed or crashed.
- [ ] **Step 4: Run** `PYTHONUTF8=1 uv run pytest -q tests/test_parsers.py` → PASS (LiteParse test 5 runs in a row).
- [ ] **Step 5: Commit** `fix: serialize pdfium and the local OCR model across threads`

---

### Task 4: ingest processes several documents at once

**Files:**
- Modify: `app/kb/ingest.py`, `app/kb/ingest_main.py`, `app/kb/services.py`, `app/config.py`, `.env.example`, `docker/.env.example`
- Test: `tests/test_ingest.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: `Ingester`, `KbServices.ingester`, `Settings.kb_ingest_*` (Task 1); `PDFIUM_LOCK` (Task 3).
- Produces: `Settings.ingest_concurrency: int = 1` from `INGEST_CONCURRENCY` (positive int, else `ValueError("INGEST_CONCURRENCY 必須是正整數")`); `Ingester.__init__(..., concurrency: int = 1)`; `Ingester.run_forever(stop)` keeps its signature and becomes the dispatcher.

- [ ] **Step 1: Write the failing tests** in `tests/test_ingest.py` (threaded ingester, real `run_forever` in a thread, `stop` set at the end):
  - `test_two_documents_are_processed_at_the_same_time`: concurrency 2; a parser whose `parse` waits on a `threading.Barrier(2, timeout=5)`; both documents end `ready` (sequential processing breaks the barrier → `BrokenBarrierError` → document failed → test fails).
  - `test_never_more_than_the_concurrency_limit`: concurrency 2, 5 documents, parser records the max number of simultaneous `parse` calls → `== 2`.
  - `test_concurrency_one_processes_in_order`: concurrency 1, 3 documents → `parse` order equals document ids.
  - `test_a_crashing_document_frees_its_slot`: concurrency 1; monkeypatch `process` to raise `RuntimeError` for the first doc only → the second doc still ends `ready`.
  - `test_deleting_one_document_leaves_the_parallel_one_alone`: concurrency 2, two docs with a slow multi-page parser; delete doc A mid-parse → A row gone, B `ready` with its chunks in the store.
  - `test_two_documents_of_one_knowledge_base_index_side_by_side`: concurrency 2, same KB → both `ready` and `InMemoryChunkStore` holds both docs' chunks.
  - `test_unavailable_parser_still_waits_with_parallel_slots`: concurrency 2, OCR parser unavailable → its doc stays `queued` while the other parser's doc completes.
  - `test_parsers_are_released_only_when_all_slots_are_idle`: concurrency 2; fake parser counts `release()` calls while another doc is still parsing → 0 until both finish.
  - `test_stop_waits_for_running_documents`: stop during a slow parse → `run_forever` returns after the in-flight doc stops at its next page; doc left `parsing` (stale requeue picks it up).
  - In `tests/test_config.py`: `INGEST_CONCURRENCY` default 1, parsing, and rejection of `0`/`abc`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** the dispatcher in `Ingester.run_forever`: a `ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="kb-ingest")`, an in-flight counter guarded by a `threading.Condition`; claim only while `in_flight < concurrency`; each submitted job runs `process(doc_id, stop)` in `try/finally` that decrements and notifies; when nothing is claimable and `in_flight == 0` run `_release_parsers()`, `cleanup()` and the periodic maintenance (same `CLEANUP_INTERVAL_SECONDS` rule as today); wait on the condition with `poll_seconds` timeout; on `stop`, stop claiming and `shutdown(wait=True)`. `wake()` notifies the condition. `threaded=False` (`_drain`) stays sequential. Pass `settings.ingest_concurrency` from `build_kb_services`.
- [ ] **Step 4: Run** `PYTHONUTF8=1 uv run pytest -q tests/test_ingest.py tests/test_config.py` → PASS; then the whole suite.
- [ ] **Step 5: Commit** `feat: ingest processes several documents at once`

---

### Task 5: LightOnOCR keeps several pages in flight

**Files:**
- Modify: `app/kb/parsers/lightonocr.py`, `app/kb/parsers/ocr_engines.py`, `app/kb/services.py`, `app/config.py`, `.env.example`, `docker/.env.example`
- Test: `tests/test_parsers.py`, `tests/test_config.py`, `tests/test_ocr_engines.py` (or wherever `VllmOcrEngine` is tested)

**Interfaces:**
- Consumes: `PDFIUM_LOCK` (Task 3).
- Produces: `Settings.ocr_concurrency: int` from `OCR_CONCURRENCY` (positive int; temporary default 2 until Task 6), `Settings.ocr_timeout_seconds: float = 600` from `OCR_TIMEOUT_SECONDS` (positive); `LightOnOcrParser(engine, concurrency: int = 1)`; `VllmOcrEngine(..., timeout: float = 600)` used for the chat-completions request (availability check keeps 3 s).

- [ ] **Step 1: Write the failing tests** (fake engine with a configurable delay and a counter of simultaneous `ocr()` calls):
  - `test_lightonocr_keeps_at_most_concurrency_pages_in_flight`: 6-page PDF, concurrency 3 → max simultaneous `ocr()` == 3.
  - `test_lightonocr_results_are_in_page_order`: engine returns page-specific grounding text with random delays → chunks ordered 1..6 with the right text and crop names `img_{page}_{i}.png`.
  - `test_lightonocr_progress_counts_every_page`: progress calls end with `(6, 6)` and are non-decreasing.
  - `test_lightonocr_one_failed_page_fails_the_document`: page 4 raises → `ParseError` 「第 4 頁 OCR 失敗」, no `ParsedChunk` list returned, no later page submitted after the failure is seen.
  - `test_stop_while_pages_are_in_flight_indexes_nothing`: `on_progress` raises a custom exception at page 2 → it propagates unchanged and the engine receives no new pages after it.
  - `test_lightonocr_holds_few_rendered_pages`: count rendered-but-not-finished pages ≤ concurrency.
  - `test_vllm_engine_uses_the_configured_timeout`: transport records the request timeout → 600 by default, configurable.
  - Config tests for `OCR_CONCURRENCY` and `OCR_TIMEOUT_SECONDS` (default, parse, reject `0`).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** in `LightOnOcrParser.parse`: render pages one at a time under `PDFIUM_LOCK`, submit `engine.ocr` to a `ThreadPoolExecutor(concurrency)`, keep a FIFO of futures, and when it holds `concurrency` futures wait for the oldest; for each finished page (in page order) run `page_markdown`, append the chunk and call `on_progress` in the parsing thread; on any exception cancel pending futures, wait for running ones, and re-raise (page errors wrapped as today). Wire `settings.ocr_concurrency` / `ocr_timeout_seconds` through `build_ocr_engine` and `build_kb_services`.
- [ ] **Step 4: Run** `PYTHONUTF8=1 uv run pytest -q` → PASS.
- [ ] **Step 5: Commit** `feat: LightOnOCR sends several pages to vLLM at once`

---

### Task 6: Docs, Docker verification and the OCR_CONCURRENCY benchmark (controller)

**Files:**
- Modify: `docs/services.md`, `README.md`, `docker/README.md`, `docs/vllm-rtx2060.md`, `app/config.py` (final `OCR_CONCURRENCY` default), `.env.example`, `docker/.env.example`

- [ ] **Step 1: Write** the ingest section of `docs/services.md` (spec §1.2): responsibilities; table of parsers and where each runs (LiteParse in ingest on CPU; LightOnOCR rendered in ingest, recognized by vLLM; text parser in ingest); waiting for unavailable parsers, heartbeat and stale requeue; `INGEST_CONCURRENCY` × `OCR_CONCURRENCY` ≤ `--max-num-seqs`; rule for where a future parser belongs (library → ingest; GPU model → model service). README settings table: `KB_INGEST_*`, `INGEST_CONCURRENCY`, `OCR_CONCURRENCY`, `OCR_TIMEOUT_SECONDS`. `docker/README.md`: upgrade notes from spec §9.
- [ ] **Step 2: Docker verification** (stop RAGFlow first if running; back up `app.db`):
  - build; `docker compose up -d --remove-orphans` (no profile) → no vllm, old `worker` container gone, LiteParse upload works, LightOnOCR shows unavailable.
  - `docker compose --profile lightonocr up -d` → LightOnOCR works.
  - `INGEST_CONCURRENCY=2`: upload two PDFs → both `parsing` at the same time (poll the DB).
  - With `INGEST_CONCURRENCY=2`, upload 4 LiteParse PDFs at once: all end `ready`, the chunk viewer shows every page, and the ingest log has no httpx / gRPC errors (shared `EmbedderClient` and `MilvusClient` used from several threads, spec §3.2).
- [ ] **Step 3: Benchmark** `test_data/草莓園入園方案.pdf` with LightOnOCR at `OCR_CONCURRENCY` 1, 2, 4 (`INGEST_CONCURRENCY=1`, `--max-num-seqs 4`): total time, per-page average, peak memory of ingest / vllm / embedder (`docker stats`). Pick the fastest value that keeps memory clear of the VM limit; set it as the default; record the table in `docs/vllm-rtx2060.md`.
- [ ] **Step 4: Run** the full backend and frontend checks; clean up test accounts.
- [ ] **Step 5: Commit** `docs: ingest service and OCR concurrency benchmark`

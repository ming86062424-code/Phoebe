# LLM 供應商切換與系統設定頁 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 聊天可在 Ollama 與 Gemini 之間切換，由管理員在系統設定頁設定供應商、模型、系統提示詞與 temperature。

**Architecture:** 新增 `app/llm/` 抽象層（`ChatClient` 協定、Ollama 與 Gemini 實作、工廠）；系統設定存在 SQLite `app_settings`，每次聊天讀取；管理員由 `ADMIN_USERNAMES` 決定；前端新增僅管理員可見的設定頁。

**Tech Stack:** FastAPI、SQLite、httpx、google-genai SDK、React + TypeScript、Vitest

**Spec:** `docs/superpowers/specs/2026-10-19-llm-providers-design.md`

## Global Constraints

- embedding 維持本機 Ollama，不得改動 `app/kb/` 的 embedding 行為
- `GEMINI_API_KEY` 只從環境變數／`.env` 讀；不得存進 DB、不得出現在任何 API 回應或 log
- 新環境變數：`GEMINI_API_KEY`（空）、`GEMINI_MODEL`（`gemini-2.5-flash`）、`ADMIN_USERNAMES`（空，逗號分隔，去空白，大小寫須完全一致）
- 設定鍵與驗證：`llm_provider` ∈ {`ollama`,`gemini`}；`ollama_model`、`gemini_model` 非空；`system_prompt` ≤ 4000 字；`temperature` 為 `null` 或 0–2
- 錯誤訊息（中文，逐字）：「無法連線到 Ollama」、「無法連線到 Gemini」、「Gemini API key 無效」、「Gemini 配額已用完或請求太頻繁，請稍後再試」、「找不到模型 <名稱>」、「未設定 GEMINI_API_KEY」、「Gemini 目前不可用：未設定 GEMINI_API_KEY」、「需要管理員權限」
- 開始串流前的 LLM 錯誤 → HTTP 503；串流中 → NDJSON `error` 事件（沿用現有機制）
- 新增依賴以 `uv add` 加入，並固定為發布超過兩週的版本
- 介面文字繁體中文；LF 換行；Windows Git Bash；`uv` 在 `/c/Users/anny3/AppData/Roaming/Python/Python312/Scripts`，Node 在 `/c/Program Files/nodejs`，`PYTHONUTF8=1`
- 每個 commit 結尾：空一行後 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

1. **管理員在使用中把供應商切到 Gemini，另一個 Gunicorn worker 仍用舊設定**：期望下一則訊息就用新設定（每次請求讀 DB，無快取）。→ Task 4 `test_settings_change_applies_to_next_request`
2. **儲存的模型名稱已從 Ollama 刪除／Gemini 下架**：期望聊天回 503「找不到模型 X」，設定頁仍顯示該名稱並標註找不到。→ Task 3 `test_gemini_404_maps_to_model_not_found`、Task 5 `shows saved model missing from list`
3. **系統提示詞與知識庫同時存在**：期望兩者都送出，不互相覆蓋。→ Task 4 `test_system_prompt_combines_with_knowledge_base`
4. **API key 外洩**：期望 `/api/settings` 與錯誤訊息都不含 key。→ Task 4 `test_settings_response_never_contains_api_key`
5. **非管理員直接開 `#/settings`**：期望被導回 `#/chat`，API 403。→ Task 2 `test_non_admin_gets_403`、Task 5 `non-admin is redirected from settings`

---

### Task 1: LLM 抽象層與 Ollama 客戶端遷移

**Files:**
- Create: `app/llm/__init__.py`、`app/llm/base.py`、`app/llm/ollama.py`
- Modify: `app/ollama_client.py`（改為從 `app.llm` 重新匯出）、`app/chat_stream.py`、`app/routes/chat.py`、`app/deps.py`、`tests/fakes.py`
- Test: `tests/test_ollama_client.py`、`tests/test_chat.py`

**Interfaces:**
- Produces:
  - `app.llm.base`：`ChatDelta`（沿用）、`LlmError`、`LlmUnavailable(message)`、`ModelNotFound(model)`（訊息「找不到模型 {model}」）、`LlmStreamError`、`LlmAuthError`、`LlmRateLimited`
  - `ChatClient`（Protocol）：`stream_chat(messages: list[dict], *, system: str | None = None, temperature: float | None = None) -> AsyncIterator[ChatDelta]`、`async list_models() -> list[str]`、`async availability() -> tuple[bool, str]`
  - `app.llm.ollama.OllamaChatClient(base_url, model, think=True, transport=None)`：`system` 放在 messages 開頭、`temperature` 放 `options.temperature`（None 不送）、`list_models` 回 `/api/tags` 名稱並排除含 `embed` 者、`availability` 以 `/api/tags` 判斷
  - `app/ollama_client.py`：`OllamaClient = OllamaChatClient`、`OllamaError = LlmError`、`OllamaUnavailable = LlmUnavailable`（預設訊息「無法連線到 Ollama」）、`OllamaStreamError = LlmStreamError`、`ModelNotFound`、`ChatDelta` 重新匯出（evals 與既有測試不需改）
  - `chat_stream.ndjson_reply` 新增參數 `system: str | None`、`temperature: float | None`，並以 `LlmError` 取代 `OllamaError` 判斷
  - `tests/fakes.FakeChatClient` 記錄 `system`、`temperature`（`calls` 改為 `[{"messages","system","temperature"}]`，同步更新使用它的測試）

- [ ] **Step 1: Write the failing tests** — `test_ollama_sends_system_and_temperature`、`test_ollama_omits_temperature_when_none`、`test_ollama_list_models_excludes_embedding_models`、`test_ollama_availability`；既有測試照常
- [ ] **Step 2: Run** `uv run pytest -q` → FAIL
- [ ] **Step 3: Implement**；行為與現有完全相同（現有測試全部通過）
- [ ] **Step 4: Run** `uv run pytest -q` → 全部通過
- [ ] **Step 5: Commit** `refactor: LLM client abstraction with Ollama implementation`

---

### Task 2: 管理員與系統設定儲存／API

**Files:**
- Create: `app/llm_settings.py`、`app/routes/settings.py`
- Modify: `app/config.py`（`admin_usernames: tuple[str, ...]`、`gemini_api_key: str`、`gemini_model: str`）、`app/auth.py`、`app/db.py`（`app_settings` 表）、`app/main.py`、`.env.example`、`docker/.env.example`
- Test: `tests/test_llm_settings.py`、`tests/test_settings_api.py`、`tests/test_auth.py`、`tests/test_config.py`

**Interfaces:**
- Consumes: `ChatClient.availability`（Task 1；Gemini 的 availability 在 Task 3 前以 key 是否存在判斷即可：本任務用 `app.llm.factory.provider_availability(settings) -> dict[str, tuple[bool, str]]`，Ollama 呼叫 `OllamaChatClient.availability()`、Gemini 回 `(bool(key), "" or "未設定 GEMINI_API_KEY")`）
- Produces:
  - `LlmSettings`（dataclass）：`llm_provider: str`、`ollama_model: str`、`gemini_model: str`、`system_prompt: str`、`temperature: float | None`
  - `load_llm_settings(conn, settings: Settings) -> LlmSettings`；`save_llm_settings(conn, settings: Settings, values: dict) -> LlmSettings`（部分更新，驗證失敗 `ValueError`）
  - `current_user` 結果加 `is_admin: bool`；`app.auth.require_admin` 依賴（403「需要管理員權限」）
  - 路由：`GET /api/settings`、`PUT /api/settings`、`GET /api/settings/models?provider=`（本任務的 models 端點只接 Ollama；Gemini 在 Task 3 接上，先回 503「未設定 GEMINI_API_KEY」或「無法連線到 Gemini」）
  - `Settings.from_env` 讀 `ADMIN_USERNAMES`、`GEMINI_API_KEY`、`GEMINI_MODEL`；`.env.example` 兩份都加上（含中文註解，key 留空）

- [ ] **Step 1: Write the failing tests**

```python
def test_defaults_come_from_env(settings_with_env)            # 無資料列 → provider ollama、ollama_model == CHAT_MODEL、gemini_model == GEMINI_MODEL、temperature None
def test_partial_update_keeps_other_keys(...)
def test_validation_rejects_bad_values(...)                    # provider "x"、temperature 2.5、-0.1、system_prompt 4001 字 → ValueError
def test_admin_usernames_parsing(monkeypatch)                  # " alice, bob ,," → ("alice","bob")
def test_me_reports_is_admin(...)                              # admin True / 其他 False
def test_non_admin_gets_403(...)                               # GET/PUT /api/settings、GET /api/settings/models
def test_put_to_unavailable_provider_is_409(...)               # 沒有 key 時切 gemini → 409「Gemini 目前不可用：未設定 GEMINI_API_KEY」
def test_get_settings_reports_provider_availability(...)
def test_settings_response_never_contains_api_key(...)         # GEMINI_API_KEY="sk-secret" → 回應文字不含 "sk-secret"
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run** `uv run pytest -q` → 全部通過
- [ ] **Step 5: Commit** `feat: admin role and system LLM settings API`

---

### Task 3: Gemini 客戶端

**Files:**
- Create: `app/llm/gemini.py`、`app/llm/factory.py`（補完）
- Modify: `pyproject.toml`、`uv.lock`（`google-genai`，發布超過兩週的版本，以 `uv add "google-genai==<版本>"` 固定）、`app/routes/settings.py`（models 端點接 Gemini）
- Test: `tests/test_gemini_client.py`

**Interfaces:**
- Consumes: Task 1 的 `ChatClient`、例外；Task 2 的 `LlmSettings`、`Settings.gemini_api_key`
- Produces:
  - `GeminiChatClient(api_key: str, model: str, think: bool, client=None)`；`client` 可注入替身（預設 `genai.Client(api_key=..., http_options=...)`，連線逾時 10 秒）
  - `stream_chat`：角色 `assistant`→`model`；`GenerateContentConfig(system_instruction=system, temperature=temperature, thinking_config=ThinkingConfig(include_thoughts=think))`；`part.thought` → thinking，其他文字 → content；空文字略過
  - 錯誤對應：`genai.errors.APIError.code` 400/403 → `LlmAuthError("Gemini API key 無效")`（400 只在訊息含 API key 時，其餘 400 → `LlmStreamError(str)`）、404 → `ModelNotFound(model)`、429 → `LlmRateLimited`、≥500 → `LlmUnavailable("無法連線到 Gemini")`；httpx 連線錯誤 → `LlmUnavailable("無法連線到 Gemini")`；串流開始後的任何上述錯誤以同一類別拋出（`chat_stream` 會轉成 `error` 事件）
  - `list_models()`：過濾 `supported_actions` 含 `generateContent` 且去掉 `models/` 前綴後以 `gemini` 開頭，排序
  - `availability()`：無 key → `(False, "未設定 GEMINI_API_KEY")`，否則 `(True, "")`
  - `app.llm.factory.build_chat_client(settings: Settings, llm: LlmSettings) -> ChatClient`；`provider_availability` 改用各客戶端的 `availability()`

- [ ] **Step 1: Write the failing tests**（替身 SDK client：`aio.models.generate_content_stream` 回傳 async iterator of fake responses；`aio.models.list` 回傳 fake models）

```python
async def test_gemini_maps_roles_and_config(...)       # contents 角色 ["user","model","user"]；config.system_instruction、temperature、thinking_config.include_thoughts
async def test_gemini_yields_thought_and_content(...)  # part.thought=True → thinking；否則 content
async def test_gemini_401_403_maps_to_auth_error(...)  # 「Gemini API key 無效」
async def test_gemini_404_maps_to_model_not_found(...) # 「找不到模型 gemini-x」
async def test_gemini_429_maps_to_rate_limited(...)
async def test_gemini_5xx_and_connect_map_to_unavailable(...)
async def test_gemini_list_models_filters_and_sorts(...)
async def test_gemini_availability_without_key(...)
def test_factory_builds_client_for_provider(...)
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run** `uv run pytest -q` → 全部通過
- [ ] **Step 5: Commit** `feat: Gemini chat client and model listing`

---

### Task 4: 聊天使用系統設定

**Files:**
- Modify: `app/deps.py`（`get_chat_client` 改為讀 `load_llm_settings` 後呼叫 `build_chat_client`）、`app/routes/chat.py`（`_with_knowledge` 回傳 `(system_text | None, sources)`；組合系統提示詞；把 `system`、`temperature` 傳給 `ndjson_reply`）、`app/kb/rag.py`（如需調整）
- Test: `tests/test_chat.py`

**Interfaces:**
- Consumes: Task 1–3
- Produces: 聊天依設定選擇供應商與模型；`system = "\n\n".join(非空的 [system_prompt, kb_prompt]) or None`

- [ ] **Step 1: Write the failing tests**

```python
def test_chat_uses_configured_provider(...)                   # 設 gemini（有 key）→ factory 建 Gemini（以替身驗證型別／模型名）
def test_settings_change_applies_to_next_request(...)         # 兩次送訊息之間 PUT 設定 → 第二次用新模型
def test_system_prompt_and_temperature_are_passed(...)        # fake.calls[-1]["system"] == "你是…"、temperature == 0.3
def test_system_prompt_combines_with_knowledge_base(...)      # system 同時含使用者提示詞與「參考資料：」
def test_no_system_when_both_empty(...)                       # system is None
def test_unavailable_provider_returns_503(...)                # 設定為 gemini 後移除 key → 503「Gemini 目前不可用：未設定 GEMINI_API_KEY」
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run** `uv run pytest -q` → 全部通過
- [ ] **Step 5: Commit** `feat: chat follows the system LLM settings`

---

### Task 5: 前端系統設定頁

**Files:**
- Create: `frontend/src/features/settings/SettingsPage.tsx`、`SettingsPage.module.css`、`SettingsPage.test.tsx`
- Modify: `frontend/src/api/types.ts`（`User.is_admin`、`LlmSettings`、`ProviderStatus`）、`frontend/src/api/client.ts`（`api.getSettings()`、`api.saveSettings(partial)`、`api.listModels(provider)`）、`frontend/src/features/registry.ts`（`adminOnly?: boolean`；新增 `{ id: "settings", label: "系統設定", path: "/settings", icon: Settings, adminOnly: true }`）、`frontend/src/shell/NavSidebar.tsx`、`frontend/src/App.tsx`（adminOnly 路由守門）
- Test: 新增與既有的 shell/App 測試

**Interfaces:**
- Consumes: Task 2–3 的 API 形狀（規格 §2.7）
- Produces: 規格 §3 的頁面行為

- [ ] **Step 1: Write the failing tests**

```tsx
it("non-admin does not see 系統設定 in the nav")
it("non-admin is redirected from settings")              // #/settings → #/chat
it("loads settings and the model list for the provider")
it("disables an unavailable provider with its reason")    // 「未設定 GEMINI_API_KEY」
it("switching provider fetches that provider's models")
it("shows saved model missing from list")                // 「（清單中找不到）」
it("model list failure shows reason and retry")
it("temperature uses default until unchecked")           // 勾選時送 null；取消後送滑桿值
it("saves and shows 已儲存; shows server error on failure")
it("asks before leaving with unsaved changes")
```

- [ ] **Step 2: Run** `npm test` → FAIL
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run** `npm test && npm run lint && npm run typecheck && npm run build` → 全部通過
- [ ] **Step 5: Commit** `feat: admin settings page for LLM provider, model, prompt and temperature`

---

### Task 6: 文件與實測

**Files:**
- Modify: `README.md`、`docker/README.md`
- Test: 手動實測

- [ ] **Step 1: 文件**：管理員設定（`ADMIN_USERNAMES`）、Gemini key（只放 `.env`／`docker/.env`，取得方式）、設定頁用法、GPU 專給 OCR 時建議用 Gemini 聊天
- [ ] **Step 2: 實測（本機 uvicorn）**：設定 `ADMIN_USERNAMES` → 管理員登入看到「系統設定」；Ollama 模型清單載入；改系統提示詞後聊天生效；若 `GEMINI_API_KEY` 存在，切到 Gemini 並串流聊天（含思考摘要）；不存在則確認選項停用
- [ ] **Step 3: Commit** `docs: admin and Gemini settings`

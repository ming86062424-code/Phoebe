# LLM 供應商切換（Ollama / Gemini）與系統設定頁 — 設計規格

- 日期：2026-10-19
- 子專案：2 / 4（部署基礎 → **LLM 供應商切換** → Milvus 混合查詢 → 多模態知識庫）
- 建立在子專案 1（`feature/deployment`，PR #6）之上
- 狀態：使用者授權直接開發（規格與計畫不需逐段確認）

> **之後的變更**（2026-10-27 補註）：以下內容是當時的設計，保留作為紀錄。後來改變的部分：
>
> - 管理員改由 `ADMIN_USERNAME`、`ADMIN_PASSWORD` 指定，啟動時自動建立帳號並同步密碼；`ADMIN_USERNAMES` 已不再使用，設定了會拒絕啟動。
> - 預設的聊天供應商改為 Gemini（`LLM_PROVIDER`，預設 `gemini`）。
> - embedding 已不使用 Ollama，改由 embedder 服務產生（`2026-10-19-multimodal-kb-design.md`）。

## 1. 目的與範圍

**目的**：聊天可以使用本機 Ollama 或雲端 Gemini，由管理員在系統設定頁切換，讓 GPU 可以專供知識庫解析（子專案 1 的 vLLM OCR）。

**需求（使用者確認）**
1. 聊天 LLM 可在 Ollama 與 Gemini 之間切換；**embedding 維持本機 Ollama**（換 embedding 會讓既有向量失效，留到子專案 3）
2. 系統設定頁，只有**管理員**能進入與修改；設定對全系統生效
3. 管理員由 `.env` 的 `ADMIN_USERNAMES`（逗號分隔）指定；沒設定時沒有管理員
4. 可設定項目：**供應商與模型**、**系統提示詞**、**temperature**
5. 模型清單**即時向 API 查詢**（Gemini：可用於產生內容的模型；Ollama：本機已下載的模型）
6. Gemini API key **只放 `.env`**（`GEMINI_API_KEY`），不進資料庫、不從介面輸入或顯示；沒設定時 Gemini 選項不可用並說明原因
7. 思考模式維持 `.env` 的 `CHAT_THINK`；兩個供應商都要能顯示思考內容（Gemini 顯示思考摘要）

**明確不做（YAGNI）**：embedding 切換、每位使用者各自的模型偏好、多組 API key、用量／費用統計（屬於監控子專案）、圖片輸入（子專案 4）、其他雲端供應商。

## 2. 架構

### 2.1 LLM 抽象層 `app/llm/`
```
app/llm/
├── base.py        # ChatDelta、ChatRequest、LlmError 階層、ChatClient 協定
├── ollama.py      # OllamaChatClient（由 app/ollama_client.py 搬入）
├── gemini.py      # GeminiChatClient（google-genai SDK）
└── factory.py     # build_chat_client(settings, llm_settings) -> ChatClient
```
- `ChatDelta(kind: "thinking"|"content", text)`：沿用
- `ChatClient` 協定：`async stream_chat(messages: list[dict], *, system: str | None, temperature: float | None) -> AsyncIterator[ChatDelta]`；`async list_models() -> list[str]`；`async availability() -> tuple[bool, str]`
  - `messages`：`[{"role": "user"|"assistant", "content": str}]`（不含 system）
- 例外：`LlmError` ← `LlmUnavailable`（連不上、未設定 key）、`ModelNotFound(model)`、`LlmStreamError`、`LlmAuthError`（key 無效）、`LlmRateLimited`（配額／速率）。訊息為中文，並標明供應商（例：「無法連線到 Ollama」、「無法連線到 Gemini」、「Gemini API key 無效」、「Gemini 配額已用完或請求太頻繁，請稍後再試」）
- `app/ollama_client.py` 保留為相容匯出（evals 與既有 import），內容改由 `app/llm/ollama.py` 提供

### 2.2 Ollama
- 行為不變；新增：`system` → 放在 messages 開頭的 `{"role":"system"}`；`temperature` → `options.temperature`（None 時不送）；`list_models()` → `GET /api/tags` 的名稱（排除名稱含 `embed` 的模型）；`availability()` → `GET /api/tags` 成功即可用

### 2.3 Gemini
- SDK：`google-genai`（固定一個發布超過兩週的版本）
- 串流：`client.aio.models.generate_content_stream(model, contents, config)`
  - `contents`：`user` → `"user"`、`assistant` → `"model"`
  - `config`：`system_instruction=system`、`temperature`、`thinking_config=ThinkingConfig(include_thoughts=CHAT_THINK)`
  - 回應的每個 part：`part.thought` 為真 → `ChatDelta("thinking", text)`；否則 → `ChatDelta("content", text)`
- `list_models()`：`client.aio.models.list()`，只保留 `supported_actions` 含 `generateContent` 且名稱以 `gemini` 開頭者，去掉 `models/` 前綴，排序
- `availability()`：沒有 `GEMINI_API_KEY` → `(False, "未設定 GEMINI_API_KEY")`；否則 `(True, "")`（不在每次呼叫時打 API）
- 錯誤對應：SDK 的 `APIError` 依 HTTP 狀態 → 400/403 金鑰問題 `LlmAuthError`、404 `ModelNotFound`、429 `LlmRateLimited`、5xx 與連線錯誤 `LlmUnavailable`；串流中途錯誤 → `LlmStreamError`
- 逾時：連線 10 秒，整體回應不設上限（與 Ollama 一致）

### 2.4 系統設定儲存
- 新資料表 `app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)`（值為 JSON）
- 設定鍵與預設值（沒有資料列時使用；預設值來自 `.env`）：

| 鍵 | 預設 | 驗證 |
|---|---|---|
| `llm_provider` | `ollama` | `ollama` 或 `gemini` |
| `ollama_model` | `CHAT_MODEL`（`gemma4:e4b`） | 非空字串 |
| `gemini_model` | `GEMINI_MODEL`（`.env`，預設 `gemini-2.5-flash`） | 非空字串 |
| `system_prompt` | 空字串 | 最多 4000 字 |
| `temperature` | `null`（使用供應商預設） | `null` 或 0–2 |

- `app/llm_settings.py`：`LlmSettings`（dataclass）、`load_llm_settings(conn, settings) -> LlmSettings`、`save_llm_settings(conn, values) -> LlmSettings`（部分更新，未提供的鍵不動）
- 每次聊天請求讀取一次（SQLite 很快；多 worker process 之間自然一致，不需快取失效）

### 2.5 系統提示詞與知識庫
- 最終送給 LLM 的 system：`system_prompt`（若非空）＋ 空行 ＋ 知識庫提示（若對話有用知識庫）；兩者都空則不送
- `_with_knowledge` 改為回傳 `(system_text | None, sources)`，不再把 system 塞進 messages

### 2.6 管理員
- `.env`：`ADMIN_USERNAMES=alice,bob`（去除空白、比對大小寫完全一致）
- `current_user` 回傳的使用者加上 `is_admin: bool`；`GET /api/me` 一併回傳
- 依賴 `require_admin`：非管理員 → 403「需要管理員權限」

### 2.7 API（皆需登入；設定相關需管理員）
| 方法 | 路徑 | 說明 |
|---|---|---|
| GET | `/api/settings` | `{llm_provider, ollama_model, gemini_model, system_prompt, temperature, providers: {ollama: {available, reason}, gemini: {available, reason}}}` |
| PUT | `/api/settings` | 部分更新；驗證失敗 422；切到不可用的供應商 → 409「Gemini 目前不可用：未設定 GEMINI_API_KEY」 |
| GET | `/api/settings/models?provider=ollama\|gemini` | `{models: [...]}`；供應商不可用或查詢失敗 → 503 與原因 |

聊天端點不變；依設定選擇供應商與模型。開始串流前的錯誤沿用 503；串流中錯誤送 `error` 事件（子專案 1 的機制）。

## 3. 前端
- `User` 型別加 `is_admin`
- 功能註冊表項目加 `adminOnly?: boolean`；`NavSidebar` 與路由都依 `useCurrentUser().is_admin` 過濾（非管理員直接打網址 → 導回 `#/chat`）
- 新功能頁「系統設定」（`/settings`，圖示 `Settings`，`adminOnly: true`）：
  - 供應商單選（Ollama / Gemini）；不可用的選項停用並顯示原因
  - 模型下拉選單：切換供應商時向 `/api/settings/models` 取得清單；載入中、失敗（顯示原因與「重試」）；目前儲存的模型若不在清單中，仍顯示並標註「（清單中找不到）」
  - 系統提示詞多行輸入（字數 / 4000）
  - temperature：「使用預設」勾選框；取消勾選後顯示 0–2、間隔 0.1 的滑桿與數值
  - 「儲存」：成功顯示「已儲存」；失敗顯示伺服器訊息；有未儲存變更時離開頁面要確認（沿用 navGuard）
- 介面文字繁體中文

## 4. 設定檔與部署
- `.env.example`（本機）與 `docker/.env.example`：新增 `GEMINI_API_KEY=`（註解說明：取得方式、只放這裡）、`GEMINI_MODEL=gemini-2.5-flash`、`ADMIN_USERNAMES=`
- `docker-compose.yml`：web、worker 經 `env_file` 取得上述變數（worker 不用 Gemini，但共用同一份設定無妨）
- README：說明管理員設定、Gemini key、設定頁

## 5. 錯誤處理
| 情境 | 行為 |
|---|---|
| 設定為 Gemini 但 key 被移除 | 聊天回 503「Gemini 目前不可用：未設定 GEMINI_API_KEY」 |
| Gemini key 無效 | 503「Gemini API key 無效」 |
| 配額用完 | 503（串流前）或 `error` 事件（串流中）「Gemini 配額已用完或請求太頻繁，請稍後再試」 |
| 模型不存在 | 503「找不到模型 <名稱>」 |
| 非管理員存取設定 API | 403 |

## 6. 測試
- 抽象層：Ollama 客戶端既有測試照常通過；system 與 temperature 正確送出；`list_models` 過濾 embed 模型
- Gemini：以假的 SDK 客戶端（注入 `genai.Client` 替身）測試：角色轉換、system_instruction、temperature、include_thoughts、thought part → thinking、錯誤對應（400/403/404/429/500/連線）、`list_models` 過濾與排序、無 key 的 availability
- 設定：預設值來自 `.env`；部分更新；驗證（provider、temperature 範圍、system_prompt 長度）；切到不可用供應商 409
- 管理員：`ADMIN_USERNAMES` 解析；`/api/me` 的 `is_admin`；非管理員 403
- 聊天：依設定選用供應商（以替身驗證）；system 合併（系統提示詞＋知識庫）
- 前端：設定頁載入、切換供應商取清單、停用不可用供應商、temperature 預設勾選、儲存成功／失敗、非管理員看不到選單項目且被導回
- 實測：Ollama 聊天照常；有 `GEMINI_API_KEY` 時切到 Gemini 串流聊天（含思考摘要）

## 7. 開發流程
- 分支 `feature/llm-providers`（從 `feature/deployment` 開出，疊在 PR #6 上）；PR 目標 `feature/deployment`，#6 合併後改為 `main`

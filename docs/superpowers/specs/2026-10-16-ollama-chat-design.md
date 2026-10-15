# Ollama 本地 AI 對話介面 — 設計規格

- 日期：2026-10-16
- 狀態：待審核

## 1. 目的與範圍

**目的**：練習 superpowers 開發流程（brainstorming → spec → plan → TDD → review），以「最小但完整」為原則。

**需求（使用者明確提出）**
- 串接本機 Ollama，聊天模型為 `gemma4:e4b`
- 多人帳號：可註冊、帳號密碼登入，每人只能看到自己的對話
- 資料庫使用 SQLite，對話紀錄持久保存，重新登入後仍在
- 回覆串流顯示（邊產生邊顯示）
- 多輪對話，記住上下文
- 後端使用 Python + FastAPI，以 uv 管理環境
- 最小版 DeepEval 品質評估，評審模型為 `qwen3.5:4b`

**明確不做（YAGNI）**
模型切換、編輯／重新產生訊息、Markdown 渲染與程式碼高亮、忘記密碼、對話自動命名、RAG、RAGAS、Langfuse、CI 上執行評估。

**後續子專案（不在本規格內）**：RAG（PDF 上傳、chunk、embedding、檢索）+ RAGAS 評估，屆時另走一次 brainstorming → spec。

## 2. 技術選型

| 項目 | 選擇 |
|---|---|
| 語言 / 環境 | Python 3.12，uv 管理（`pyproject.toml` + `uv.lock`） |
| Web 框架 | FastAPI + uvicorn |
| 呼叫 Ollama | `httpx.AsyncClient`，串流讀取 `/api/chat` 的 NDJSON |
| 資料庫 | Python 內建 `sqlite3` |
| 密碼雜湊 | `bcrypt` |
| 前端 | 原生 HTML / CSS / JS，由 FastAPI 以靜態檔提供，無打包工具 |
| 測試 | `pytest`（單元／整合）、`deepeval`（品質評估） |

前置作業：本機尚未安裝 uv，實作第一步為安裝 uv。

## 3. 架構與目錄結構

```
Phoebe/
├── pyproject.toml
├── app/
│   ├── main.py             # 建立 FastAPI app、掛路由與靜態檔
│   ├── config.py           # 設定（環境變數可覆寫）
│   ├── db.py               # SQLite 連線、建表
│   ├── auth.py             # 註冊、登入、登出、密碼雜湊、current_user 依賴
│   ├── conversations.py    # 對話與訊息 CRUD（所有查詢皆帶 user_id）
│   ├── ollama_client.py    # 唯一知道 Ollama 的模組
│   └── routes/             # auth、conversations、chat 路由
├── static/                 # index.html、login.html、app.js、style.css
├── tests/                  # pytest，Ollama 一律用替身
└── evals/                  # DeepEval 品質評估，需真實 Ollama，手動執行
```

**設定值**（`config.py`，皆可用環境變數覆寫）

| 名稱 | 預設值 |
|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` |
| `CHAT_MODEL` | `gemma4:e4b` |
| `EVAL_JUDGE_MODEL` | `qwen3.5:4b` |
| `DB_PATH` | `data/app.db` |
| `SESSION_DAYS` | `7` |

**單元職責**
- `ollama_client`：介面為 `stream_chat(messages: list[dict]) -> AsyncIterator[str]`，輸入完整訊息列表，輸出文字片段。以 FastAPI 依賴注入提供，測試時可替換。
- `auth`：伺服器端 session。登入產生隨機 token（32 bytes）存入 `sessions` 表與 HttpOnly、SameSite=Lax 的 cookie；登出刪除該列，立即失效。
- `conversations`：每個查詢都限定 `user_id`，存取他人資源一律視為不存在。

**多輪上下文**：每次送出訊息時，後端從 DB 讀取該對話全部歷史 + 新訊息送給 Ollama；回覆完成後存回 DB。上下文完全由 DB 保存。

**介面**：左側對話列表（新增、切換、刪除），右側聊天區；另有登入／註冊頁。對話標題取第一則使用者訊息的前 30 字。

## 4. 資料模型

```
users          id INTEGER PK, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at
sessions       token TEXT PK, user_id → users, created_at, expires_at
conversations  id INTEGER PK, user_id → users, title TEXT, created_at, updated_at
messages       id INTEGER PK, conversation_id → conversations ON DELETE CASCADE,
               role TEXT CHECK (role IN ('user','assistant')), content TEXT, created_at
```

SQLite 連線需開啟 `PRAGMA foreign_keys = ON`，CASCADE 才會生效。

## 5. API

除 register / login 外皆需登入，未登入回 401。

| 方法 | 路徑 | 說明 |
|---|---|---|
| POST | `/api/auth/register` | `{username, password}`；帳號 3–32 字元、密碼 ≥ 8 字元；帳號重複 409；成功 201 |
| POST | `/api/auth/login` | 成功設 cookie；帳號或密碼錯誤一律 401，不區分原因 |
| POST | `/api/auth/logout` | 刪 session、清 cookie |
| GET | `/api/me` | 目前使用者 `{id, username}` |
| GET | `/api/conversations` | 自己的對話列表，依 `updated_at` 新到舊 |
| POST | `/api/conversations` | 建立空對話 |
| GET | `/api/conversations/{id}` | 對話與訊息；非本人 → 404 |
| DELETE | `/api/conversations/{id}` | 刪除對話（訊息連帶刪除）；非本人 → 404 |
| POST | `/api/conversations/{id}/messages` | `{content}`；回應為串流純文字（`text/plain; charset=utf-8`） |

**串流流程**（`POST .../messages`）
1. 驗證對話屬於本人；存入 user 訊息；若為第一則訊息，設定標題
2. 讀取完整歷史，呼叫 `ollama_client.stream_chat`
3. 每收到片段即送出給瀏覽器，同時累加
4. 串流正常結束後，將完整內容存成 assistant 訊息，並更新對話 `updated_at`

## 6. 錯誤處理

| 情境 | 行為 |
|---|---|
| Ollama 連不上（串流開始前） | 回 503 `{"detail": "無法連線到 Ollama"}`；user 訊息保留，使用者可重送 |
| 模型不存在（Ollama 回 404） | 回 503，訊息中包含模型名稱 |
| 串流中途中斷（Ollama 錯誤或客戶端斷線） | 已產生的部分不存入 DB；前端顯示「回覆中斷」 |
| 訊息空白或超過 8000 字 | 422 |
| session 不存在或已過期 | 401；前端導回登入頁 |

為了在串流開始前就能回 503，路由會先取得第一個片段再開始回應；連線錯誤在這一步以例外形式拋出。

## 7. 測試策略（pytest + TDD）

- 每個測試使用獨立的暫存 SQLite 檔
- `ollama_client` 以 `app.dependency_overrides` 換成替身：固定吐出 `["你", "好"]`，或拋出連線錯誤
- 涵蓋範圍：
  - auth：註冊、登入成功、帳號重複、密碼錯誤、登出後失效、session 過期
  - 權限隔離：A 讀取或刪除 B 的對話 → 404
  - 對話：建立、列表排序、刪除時訊息一併刪除
  - 聊天：串流內容正確、回覆已存入 DB、第二輪送給 Ollama 的訊息包含完整歷史、Ollama 失敗 → 503 且不存 assistant 訊息
- `ollama_client` 本身以 `httpx.MockTransport` 模擬 NDJSON 串流，測試解析邏輯
- 前端不寫自動化測試；最後以真實 Ollama + `gemma4:e4b` 手動走一遍：註冊 → 聊天 → 多輪 → 登出再登入，確認紀錄仍在

執行：`uv run pytest tests`

## 8. DeepEval 品質評估（最小版）

**定位**：檢查「模型回答得好不好」，與檢查「程式邏輯對不對」的 `tests/` 互補。結果用於觀察與比較，不作為必須通過的關卡。

**流程**：題目 → `app.ollama_client` 呼叫真實 `gemma4:e4b` 產生回答 → 評審模型 `qwen3.5:4b`（DeepEval `OllamaModel`）依指標打 0–1 分 → 分數 ≥ 門檻 0.7 為通過。評審與受測模型不同，避免自評偏高。

**評估題目**（`evals/test_chat_quality.py`）

| 題目 | 指標 | 檢查 |
|---|---|---|
| 一般問答（例：台北下雨天景點） | `AnswerRelevancyMetric` | 回答切題 |
| 多輪記憶（先說養柴犬，再問狗的品種） | `GEval` | 記得前文 |
| 以繁體中文提問 | `GEval` | 以繁體中文回答 |
| 挑釁／引戰提問 | `ToxicityMetric` | 回答不含不當內容 |

**執行方式**
- `uv run pytest evals`，手動執行；`uv run pytest tests` 不會執行到評估
- 前提：本機 Ollama 執行中，且已有 `gemma4:e4b` 與 `qwen3.5:4b`
- 設定 `DEEPEVAL_TELEMETRY_OPT_OUT=YES`，不傳資料到 Confident AI 雲端
- 不放進 GitHub Actions（雲端 runner 連不到本機 Ollama）

**已知限制**
- 回答與評分皆有隨機性，同一題結果可能不同；4B 評審穩定度有限
- 執行時間以分鐘計

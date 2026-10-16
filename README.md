# Phoebe

在本機 Ollama 上運作的聊天應用（FastAPI + SQLite + React）。

## 事前準備

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Node 22 以上（前端建置與開發用）
- Ollama 已啟動，位於 http://localhost:11434
- 下載模型：`ollama pull gemma4:e4b`（評測另需 `ollama pull qwen3.5:4b`）

## 安裝與執行

```
uv sync
uv run uvicorn app.main:create_app --factory --reload
```

前端需先建置才能由後端提供（見下方「前端」一節），再開啟 http://localhost:8000。

## 前端

前端位於 `frontend/`（React + TypeScript + Vite）。

安裝套件：

```
cd frontend && npm install
```

開發模式（兩個終端機各跑一個指令，開啟 http://localhost:5173）：

```
uv run uvicorn app.main:create_app --factory --reload
cd frontend && npm run dev
```

正式模式（建置後只需開 uvicorn，開啟 http://localhost:8000）：

```
cd frontend && npm run build
uv run uvicorn app.main:create_app --factory
```

前端測試與檢查：

```
cd frontend && npm test
cd frontend && npm run lint
```

## 測試

單元測試：

```
uv run pytest
```

評測（僅手動執行；需要真實的 Ollama，會跑數分鐘；分數僅供觀察，不是通過／失敗的門檻）：

```
uv run --group eval pytest evals -v
```

注意：單純執行 `uv sync` 會移除 eval 群組的套件；`uv run --group eval` 會重新安裝。

## 環境變數

| 變數 | 預設值 | 說明 |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 位址 |
| `CHAT_MODEL` | `gemma4:e4b` | 聊天使用的模型 |
| `EVAL_JUDGE_MODEL` | `qwen3.5:4b` | 評測的裁判模型 |
| `DB_PATH` | `<專案根目錄>/data/app.db` | SQLite 資料庫路徑 |
| `SESSION_DAYS` | `7` | 登入工作階段有效天數 |
| `CHAT_THINK` | `true` | 是否讓模型輸出思考過程（`false` 或 `0` 為關閉） |
| `FRONTEND_DIST` | `<專案根目錄>/frontend/dist` | 前端建置輸出資料夾，由後端提供 |

## 資料位置

資料存放在 SQLite 檔案 `data/app.db`（位於專案根目錄，與啟動目錄無關；可用 `DB_PATH` 覆寫）。`data/` 已被 git 忽略。

## 專案結構

- `app/` — 後端（設定、資料庫、驗證、對話、Ollama 客戶端、路由）
- `frontend/` — 前端（React + TypeScript + Vite）
- `tests/` — 單元測試
- `evals/` — 聊天品質評測（deepeval）

## 開發流程

所有變更都透過 feature 分支與 Pull Request 進行，由 repo 擁有者審查後合併，不直接推送到 `main`。

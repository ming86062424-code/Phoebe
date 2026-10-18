# Phoebe

在本機 Ollama 上運作的聊天應用（FastAPI + SQLite + React），附 PDF 知識庫（LlamaIndex + Chroma）。

## 事前準備

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Node 22 以上（前端建置與開發用）
- Ollama 已啟動，位於 http://localhost:11434
- 下載模型：`ollama pull gemma4:e4b`（評測另需 `ollama pull qwen3.5:4b`）
- 知識庫 Embedding 模型：Ollama 中需有 `embeddinggemma-2`

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

注意：後端在啟動時才決定要不要提供 `frontend/dist`，所以第一次執行 `npm run build` 之後，必須重新啟動 uvicorn。

前端測試與檢查：

```
cd frontend && npm test
cd frontend && npm run lint
cd frontend && npm run typecheck
```

## 知識庫

側邊欄「知識庫」可建立知識庫、上傳 PDF；AI 對話上方的下拉選單可為每個對話選用一個知識庫，回答下方會列出引用來源（檔名與頁碼）。

上傳時可選擇 PDF 解析器：

| 解析器 | 適合 | 說明 |
| --- | --- | --- |
| LiteParse | 文字型 PDF | 預設、速度快、不需 GPU；掃描頁會用內建 Tesseract（`chi_tra+eng`）辨識，但中文品質有限 |
| LightOnOCR-3-0.8B | 掃描檔、表格、複雜版面 | 以 transformers 在本機 GPU 執行，輸出 Markdown；RTX 2060 上每頁約 1–2 分鐘 |

LightOnOCR 需要額外的 `ocr` 套件群組（torch CUDA 12.6、transformers，約 3GB），並以該群組啟動伺服器：

```
uv sync --group ocr
uv run --group ocr uvicorn app.main:create_app --factory
```

未以 `--group ocr` 啟動時，介面會把 LightOnOCR 顯示為「不可用」。第一次使用會從 Hugging Face 下載模型（約 1.8GB）。OCR 開始前會請 Ollama 卸載常駐模型以騰出 VRAM，處理佇列清空後會釋放 OCR 模型。

手動驗證 OCR：

```
uv run --group ocr python -m scripts.ocr_smoke 路徑/檔案.pdf
```

文件在背景依序處理（排隊中 → 解析中 → 建立索引中 → 完成／失敗）；伺服器重啟後，未完成的文件會自動重新處理。

## 測試

單元測試：

```
uv run pytest
```

評測（僅手動執行；需要真實的 Ollama，會跑數分鐘；分數僅供觀察，不是通過／失敗的門檻）：

```
uv run --group eval pytest evals -v
```

注意：單純執行 `uv sync` 會移除 eval、ocr 群組的套件；`uv run --group eval`／`uv run --group ocr` 會重新安裝。

## 環境變數

| 變數 | 預設值 | 說明 |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 位址 |
| `CHAT_MODEL` | `gemma4:e4b` | 聊天使用的模型 |
| `EVAL_JUDGE_MODEL` | `qwen3.5:4b` | 評測的裁判模型 |
| `DB_PATH` | `<專案根目錄>/data/app.db` | SQLite 資料庫路徑 |
| `SESSION_DAYS` | `7` | 登入工作階段有效天數 |
| `CHAT_THINK` | `true` | 是否讓模型輸出思考過程（`false`、`0`、`no`、`off` 為關閉，不分大小寫） |
| `EMBED_MODEL` | `embeddinggemma-2` | 知識庫 Embedding 模型（Ollama）；更換後需重新上傳文件 |
| `KB_DIR` | `<專案根目錄>/data/kb` | 上傳的 PDF 存放位置 |
| `CHROMA_DIR` | `<專案根目錄>/data/chroma` | 向量資料庫位置 |
| `KB_TOP_K` | `5` | 每次回答檢索的段落數 |
| `MAX_UPLOAD_MB` | `50` | 單一 PDF 上限 |
| `OCR_MODEL` | `lightonai/LightOnOCR-3-0.8B` | LightOnOCR 模型（Hugging Face） |
| `LITEPARSE_OCR_LANGUAGE` | `chi_tra+eng` | LiteParse 內建 Tesseract 的語言 |
| `FRONTEND_DIST` | `<專案根目錄>/frontend/dist` | 前端建置輸出資料夾，由後端提供；使用相對路徑時，會以啟動 uvicorn 的工作目錄為基準 |

## 資料位置

資料存放在 SQLite 檔案 `data/app.db`，知識庫檔案與向量分別在 `data/kb/`、`data/chroma/`（位於專案根目錄，與啟動目錄無關；可用 `DB_PATH` 覆寫）。`data/` 已被 git 忽略。

## 專案結構

- `app/` — 後端（設定、資料庫、驗證、對話、Ollama 客戶端、路由）
- `app/kb/` — 知識庫（資料存取、解析器、LlamaIndex 索引、背景處理）
- `scripts/` — 手動驗證腳本
- `frontend/` — 前端（React + TypeScript + Vite）
- `tests/` — 單元測試
- `evals/` — 聊天品質評測（deepeval）

## 開發流程

所有變更都透過 feature 分支與 Pull Request 進行，由 repo 擁有者審查後合併，不直接推送到 `main`。

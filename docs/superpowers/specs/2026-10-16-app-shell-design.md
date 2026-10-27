# 應用程式骨架（側邊欄）與聊天體驗改善 — 設計規格

- 日期：2026-10-16
- 狀態：待審核
- 前一版規格：`docs/superpowers/specs/2026-10-16-ollama-chat-design.md`（本規格只描述改動；沒提到的部分沿用前一版）

> **之後的變更**（2026-10-27 補註）：以下內容是當時的設計，保留作為紀錄。後來改變的部分：
>
> - 已實作完成（狀態「待審核」是撰寫當時的狀態）。
> - 功能選單之後陸續加入「知識庫」「系統設定」「監控」，後兩者只限管理員。

## 1. 目的與範圍

**目的**：把單頁聊天介面改成可擴充的多功能應用骨架，並改善聊天時「看起來像卡住」的體驗。同時練習較嚴謹的多人開發流程（功能分支 + PR 審核）。

**需求（使用者確認）**
1. **兩欄式骨架**：最左側是功能選單；功能頁自己決定是否有次欄（AI 對話頁有對話列表）
2. **兩個側邊欄都能收合**：功能選單收合成只有圖示的窄條（仍可切換功能）；對話列表收合後完全隱藏，只留展開鈕；收合狀態存在瀏覽器，重新整理後維持
3. **功能選單只列出已完成的功能**；目前只有「AI 對話」
4. **顯示思考過程**：思考時即時顯示思考文字與「思考中…」；開始回答後自動收合成「已思考 N 秒」，可點開。思考內容只在當下顯示，**不存 DB、不進歷史**
5. **送出後輸入框立即清空**
6. **失敗時在訊息下方顯示「重新送出」**，重新產生回答而**不新增重複的使用者訊息**
7. **四周留白、卡片式**：介面與視窗四邊間距約 16–24px，圓角加陰影；訊息區最大寬度約 800px 並置中
8. **只做電腦版**；視窗寬度 < 1024px 時對話列表預設收合，可手動展開（不寫入偏好）
9. **網址記住位置**：例如 `/#/chat/12`，重新整理後回到同一功能、同一對話
10. 登入頁採用同樣的留白與卡片風格；介面文字為繁體中文

**明確不做（YAGNI）**：手機版排版、思考內容持久化、知識庫／語音辨識／翻譯／生圖／監控等後續功能（各自另立規格）、全域狀態管理套件、Markdown 渲染。

## 2. 技術選型（新增）

| 項目 | 選擇 |
|---|---|
| 前端框架 | React + TypeScript，以 Vite 建置 |
| 路由 | `react-router` 的 HashRouter |
| 樣式 | CSS Modules + 一份全域 CSS 變數（顏色、間距、圓角） |
| 圖示 | `lucide-react` |
| 前端測試 | Vitest + Testing Library（jsdom） |
| 型別與 lint | `tsc --noEmit`、ESLint |
| Node.js | 升級至 22 LTS（目前 18.17.1 已停止支援，不符合新版 Vite 的需求）；升級方式於實作前與使用者確認 |

後端技術不變（FastAPI、SQLite、httpx、uv）。

## 3. 前端架構

```
frontend/
├── package.json
├── vite.config.ts         # dev server :5173，/api proxy 到 http://localhost:8000；build 輸出 frontend/dist
├── index.html
└── src/
    ├── main.tsx
    ├── App.tsx            # 路由：#/login、#/chat、#/chat/:id；未登入導向 #/login
    ├── api/
    │   ├── client.ts      # fetch 包裝：401 導向 #/login；把 detail（字串或 422 陣列）轉成文字
    │   ├── types.ts       # User、Conversation、Message、StreamEvent 等型別
    │   └── stream.ts      # 解析 NDJSON 回應本體 → AsyncIterable<StreamEvent>
    ├── shell/
    │   ├── AppShell.tsx   # 留白、卡片外框、功能選單 + 功能頁
    │   └── NavSidebar.tsx # 依註冊表產生選單；收合為圖示窄條；底部顯示使用者名稱與登出
    ├── features/
    │   ├── registry.ts    # 功能註冊表
    │   └── chat/
    │       ├── ChatPage.tsx
    │       ├── ConversationList.tsx
    │       ├── MessageList.tsx
    │       ├── MessageBubble.tsx
    │       ├── ThinkingBlock.tsx
    │       ├── Composer.tsx
    │       └── useChat.ts
    └── pages/LoginPage.tsx
```

**功能註冊表**（`registry.ts`）：每筆為 `{ id, label, icon, path, element }`。新增功能 = 新增一筆，選單與路由自動產生。目前唯一一筆：`{ id: "chat", label: "AI 對話", path: "/chat" }`。

**收合狀態**：存在 `localStorage`，鍵名 `phoebe.nav.collapsed`、`phoebe.chat.listCollapsed`。讀寫都包在 try/catch 中，失敗時使用預設值（皆展開）。視窗寬度 < 1024px 時對話列表預設收合，可手動展開（不寫入偏好）。

**狀態管理**：React 內建 state 與自訂 hook，不使用全域狀態套件。

**登入頁**：整合進 React（`#/login`），刪除 `static/login.html`、`static/login.js`；登入／註冊兩個表單切換的行為沿用前一版。

**安全**：所有使用者與模型文字以 React 文字節點呈現；**禁止** `dangerouslySetInnerHTML`；泡泡使用 `white-space: pre-wrap`。

**開發與執行**
- 開發：`npm run dev`（:5173）+ `uv run uvicorn app.main:create_app --factory --reload`（:8000），瀏覽器開 :5173
- 正式：`npm run build` 後只需啟動 uvicorn，開 http://localhost:8000

**資料夾命名限制**：`.gitignore` 的 Python 範本忽略 `lib/`、`dist/`、`build/` 等資料夾，前端原始碼不得使用這些資料夾名稱（`frontend/dist` 被忽略是預期行為）。`.gitignore` 需新增 `node_modules/`。

## 4. 後端改動

### 4.1 Ollama 客戶端
- 介面改為 `stream_chat(messages) -> AsyncIterator[ChatDelta]`，`ChatDelta` 具有 `kind: Literal["thinking", "content"]` 與 `text: str`
- 請求本體加上 `"think": settings.chat_think`；解析 `message.thinking` 與 `message.content`，各自非空時產出
- 新設定 `CHAT_THINK`，預設 `true`（環境變數 `CHAT_THINK=false` 關閉）
- 其他行為（例外類別、逾時、404 對應 `ModelNotFound`）不變；模型不支援思考時 Ollama 回 400，依現有規則以 `OllamaStreamError` 回報

### 4.2 串流格式：NDJSON
兩個串流端點的回應 `media_type` 為 `application/x-ndjson`，每行一個 JSON 物件：

| 事件 | 格式 | 說明 |
|---|---|---|
| thinking | `{"type":"thinking","text":"…"}` | 0 到多個 |
| content | `{"type":"content","text":"…"}` | 0 到多個 |
| done | `{"type":"done","message_id":42}` | 正常結束；assistant 訊息已存，附其 id |
| error | `{"type":"error","detail":"…"}` | 串流中途失敗；不存 assistant 訊息；為最後一個事件 |

JSON 以 `ensure_ascii=False` 輸出，每行以 `\n` 結尾。

### 4.3 端點
**`POST /api/conversations/{id}/messages`**（body `{content}`，驗證規則不變）
1. 對話不屬於本人 → 404
2. 存 user 訊息、設定標題、讀取歷史
3. 以共用串流函式回應

**`POST /api/conversations/{id}/regenerate`**（無 body）
1. 對話不屬於本人 → 404
2. 最後一則訊息不是 `user`（含對話沒有訊息）→ 409 `{"detail": "沒有需要重新產生的回答"}`
3. 讀取歷史，以共用串流函式回應

**共用串流函式**
- 先取得第一個 `ChatDelta`：失敗規則沿用前一版（`OllamaUnavailable` → 503「無法連線到 Ollama」；`ModelNotFound` 等其他 `OllamaError` 或 `httpx.HTTPError` → 503 `str(e)`；串流開始前若直接拋出原始 `httpx.HTTPError`，同樣對應 503「無法連線到 Ollama」（裁定 R4））
- 之後逐一輸出 thinking／content 事件；只累加 content
- 正常結束：另開 DB 連線存 assistant 訊息（只有 content），輸出 `done`
- 中途發生 `OllamaError` 或 `httpx.HTTPError`：不存，輸出 `error` 事件（detail 為 `str(e)`，空字串時用「回覆中斷」）後正常結束
- 客戶端斷線：不存（沿用前一版）

### 4.4 靜態檔
- 不再掛載 `static/`（刪除該資料夾），改掛載 `FRONTEND_DIST`
- 新設定 `FRONTEND_DIST`，預設 `<專案根目錄>/frontend/dist`
- `FRONTEND_DIST` 不存在或沒有 `index.html` 時：API 正常；`GET /` 回 200 純文字提示「前端尚未建置，請在 frontend/ 執行 npm run build」
- 掛載順序：所有 API router 之後

**不變**：DB schema、登入與 session、對話增刪查 API。

## 5. 聊天互動流程（`useChat`）

1. Enter 送出 → 輸入框立刻清空；立即顯示 user 泡泡與空的 assistant 泡泡；串流期間停用送出
2. 收到 `thinking` → assistant 泡泡上方顯示 `ThinkingBlock`：展開、標題「思考中…」加秒數計時、思考文字即時附加（較淡、較小字）
3. 收到第一個 `content` → `ThinkingBlock` 自動收合，標題改為「已思考 N 秒」（N 為整數秒），可點擊展開／收合；回答文字即時附加
4. 收到 `done` → 結束串流，重新載入對話列表
5. 失敗（HTTP 非 200、收到 `error` 事件、網路錯誤、串流結束卻沒有 `done`）→ 泡泡顯示錯誤文字；已收到的回答保留並加註「（回覆中斷）」；泡泡下方出現「重新送出」按鈕
6. 按「重新送出」→ 清空該泡泡，呼叫 `regenerate` 重新串流（流程同 2–5）
7. 開啟一個「最後一則是 user、沒有回答」的對話時，在最後顯示「重新送出」按鈕
8. 串流期間：不能切換、新增、刪除對話；點功能選單時先確認「回覆產生中，確定要離開？」（使用頁面內的確認元件，不使用 `window.confirm`）
9. Shift+Enter 換行；輸入法組字中（`isComposing`）按 Enter 不送出

## 6. 錯誤處理（前端）

| 情境 | 行為 |
|---|---|
| 任一 API 回 401 | 導向 `#/login` |
| 503（Ollama 連不上、模型不存在） | 泡泡顯示 detail，出現「重新送出」 |
| 409（regenerate 時已有回答） | 重新載入該對話 |
| 串流中途 `error` 事件 | 保留已收到的回答 +「（回覆中斷）」+ detail，出現「重新送出」 |
| 網路錯誤 | 泡泡顯示「無法連線到伺服器」，出現「重新送出」 |

## 7. 測試策略

**後端（pytest）**
- `FakeChatClient` 改為吐出 `ChatDelta`（可混合 thinking 與 content）
- 聊天測試改為解析 NDJSON，檢查事件順序與內容；`done` 的 `message_id` 對應 DB 中的 assistant 訊息
- 存進 DB 與送給模型的歷史不含思考內容
- 中途失敗：回應最後一個事件是 `error`、DB 沒有 assistant 訊息
- regenerate：成功、404（他人對話）、409（已有回答、沒有訊息）、503、不新增 user 訊息
- `ollama_client`：解析 `thinking` 欄位、請求帶 `think`
- 靜態檔：dist 存在時提供 index.html；不存在時 `/` 回提示、`/api/me` 仍為 401

**前端（Vitest + Testing Library）**
- `stream.ts`：一行 JSON 被切在兩個 chunk、中文字位元組被切開、最後一行沒有換行
- `useChat`：以假 fetch 模擬事件序列——thinking → content → done；error 事件；HTTP 503；串流沒有 done 就結束——檢查狀態與「重新送出」
- 元件：送出後輸入框清空；ThinkingBlock 收合與展開；收合狀態寫入 localStorage；選單由註冊表產生；未登入導向 `#/login`
- 原始碼中不得出現 `dangerouslySetInnerHTML`

**其他**
- `tsc --noEmit` 與 ESLint 無錯誤
- DeepEval：`generate_reply` 改為只收集 content；評估題目與門檻不變
- 手動驗證：完成後由使用者依檢查清單在瀏覽器走一遍（新版 README 需說明開發與正式的啟動方式）

## 8. 開發流程

- 分支：`feature/app-shell`（從 `main` 開出）
- 完成後以 PR 合併至 `main`，由使用者審核
- 實作於 PR #1（`feature/ollama-chat`）合併後開始，以免疊 PR

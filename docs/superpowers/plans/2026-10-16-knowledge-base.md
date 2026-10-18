# 知識庫 — 實作計畫

規格：`docs/superpowers/specs/2026-10-16-knowledge-base-design.md`

1. 相依套件：llama-index-core、embeddings-ollama、vector-stores-chroma、chromadb、liteparse、pypdfium2、python-multipart；`ocr` 群組（torch cu126、transformers）
2. 設定與資料庫：Settings 新增 embed_model、kb_dir、chroma_dir、kb_top_k、max_upload_mb、ocr_model；新表與欄位遷移（測試先行）
3. `app/kb/store.py` CRUD（測試先行）
4. 解析器：base、liteparse（以產生的 PDF 測試）、lightonocr（延遲載入、release）
5. `app/kb/index.py`：Chroma + LlamaIndex 寫入／刪除／檢索（MockEmbedding 測試）
6. `app/kb/worker.py` + `services.py`：同步／執行緒模式、重新排入、失敗處理（測試先行）
7. `app/routes/kb.py`：API 與權限（測試先行）
8. 聊天整合：PATCH 對話 kb_id、sources 事件與存檔（測試先行）
9. 前端：型別與 API、知識庫頁、聊天知識庫選擇與來源顯示（Vitest）
10. OCR 手動驗證腳本、README 更新、實機端到端驗證、PR

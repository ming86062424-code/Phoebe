# syntax=docker/dockerfile:1
# build context 為專案根目錄（docker-compose.yml 設 context: ..）：cd docker && docker compose build

# ---- 前端建置 ----
FROM node:24.11.1-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- 共用基底：Python 與不含 torch 的依賴 ----
FROM python:3.12.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

# LiteParse 是 Rust 原生擴充套件（內含 pdfium 與 Tesseract），執行時不需要 Node 或 tesseract 套件；
# 只補 OpenMP 執行庫（原生擴充套件常見的動態相依）。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    HOME=/home/phoebe \
    HF_HOME=/hf-cache

# 非 root 使用者；/db（SQLite，named volume）首次掛載時會沿用此權限；/data/kb 由 compose 掛載主機目錄
RUN useradd --uid 1000 --create-home --home-dir /home/phoebe phoebe \
    && mkdir -p /db /data/kb /hf-cache /app \
    && chown phoebe:phoebe /db /data /data/kb /hf-cache /app

WORKDIR /app
COPY pyproject.toml uv.lock ./
# uv 快取放在 BuildKit cache mount，不會進入任何映像層（`docker builder prune` 可清除）
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    UV_CACHE_DIR=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

# ---- web：Gunicorn，不含 torch ----
FROM base AS web
# 程式碼唯讀即可（執行時只寫 /db、/data/kb、/hf-cache 與家目錄快取），不需要遞迴 chown 產生額外映像層
COPY --chown=phoebe:phoebe app ./app
COPY --chown=phoebe:phoebe docker/gunicorn.conf.py docker/gunicorn.conf.py
COPY --chown=phoebe:phoebe --from=frontend /frontend/dist ./frontend/dist
USER phoebe
ENV KB_INGEST_MODE=external \
    DB_PATH=/db/app.db \
    KB_DIR=/data/kb
EXPOSE 8000
CMD ["gunicorn", "-c", "docker/gunicorn.conf.py", "app.main:create_app()"]

# ---- ingest：只做文件解析與索引；OCR 交給 vllm 服務，不含 torch ----
FROM base AS ingest
COPY --chown=phoebe:phoebe app ./app

# LiteParse 的 Tesseract 語言檔：預設 chi_tra+eng（含直排 chi_tra_vert）內建於映像，不必在首次 OCR 時下載。
# 來源固定在 tessdata_best 的指定 commit（2026-10-18 的 main）；其他語言仍會在首次使用時下載。
ARG TESSDATA_REV=e12c65a915945e4c28e237a9b52bc4a8f39a0cec
# Tesseract 語言檔目錄要能在執行時寫入（其他語言首次使用時下載）；
# 只 chown 這幾個（空的）目錄，不遞迴處理整個家目錄
RUN mkdir -p /home/phoebe/.tesseract-rs/tessdata \
    && chown phoebe:phoebe /home/phoebe/.tesseract-rs /home/phoebe/.tesseract-rs/tessdata
ADD --chown=phoebe:phoebe --chmod=644 \
    https://github.com/tesseract-ocr/tessdata_best/raw/${TESSDATA_REV}/chi_tra.traineddata \
    /home/phoebe/.tesseract-rs/tessdata/chi_tra.traineddata
ADD --chown=phoebe:phoebe --chmod=644 \
    https://github.com/tesseract-ocr/tessdata_best/raw/${TESSDATA_REV}/chi_tra_vert.traineddata \
    /home/phoebe/.tesseract-rs/tessdata/chi_tra_vert.traineddata
ADD --chown=phoebe:phoebe --chmod=644 \
    https://github.com/tesseract-ocr/tessdata_best/raw/${TESSDATA_REV}/eng.traineddata \
    /home/phoebe/.tesseract-rs/tessdata/eng.traineddata

USER phoebe
ENV DB_PATH=/db/app.db \
    KB_DIR=/data/kb
CMD ["python", "-m", "app.kb.ingest_main"]

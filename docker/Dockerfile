# syntax=docker/dockerfile:1
# build context 為專案根目錄：docker compose -f docker/compose.yml build

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

# 非 root 使用者；/data 與 /hf-cache 先建立並交給該使用者，named volume 首次掛載時會沿用此權限
RUN useradd --uid 1000 --create-home --home-dir /home/phoebe phoebe \
    && mkdir -p /data /hf-cache /app \
    && chown phoebe:phoebe /data /hf-cache /app

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ---- web：Gunicorn，不含 torch ----
FROM base AS web
COPY app ./app
COPY docker/gunicorn.conf.py docker/gunicorn.conf.py
COPY --from=frontend /frontend/dist ./frontend/dist
RUN chown -R phoebe:phoebe /app
USER phoebe
ENV KB_WORKER_MODE=external \
    DB_PATH=/data/app.db \
    KB_DIR=/data/kb
EXPOSE 8000
CMD ["gunicorn", "-c", "docker/gunicorn.conf.py", "app.main:create_app()"]

# ---- worker：加上 ocr 依賴（CUDA 版 torch、transformers）----
FROM base AS worker
RUN uv sync --frozen --no-dev --group ocr --no-install-project
COPY app ./app
RUN chown -R phoebe:phoebe /app
USER phoebe
ENV DB_PATH=/data/app.db \
    KB_DIR=/data/kb
CMD ["python", "-m", "app.kb.worker_main"]

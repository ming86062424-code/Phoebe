# syntax=docker/dockerfile:1
# Embedding 服務（sentence-transformers、CPU float32）。build context 為專案根目錄：
#   cd docker && docker compose build embedder
FROM python:3.12.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

# torch CPU 需要 OpenMP 執行庫
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    HOME=/home/embedder \
    HF_HOME=/hf-cache

# 非 root 使用者；/hf-cache 先建立並交給該使用者，named volume 首次掛載時會沿用此權限
RUN useradd --uid 1000 --create-home --home-dir /home/embedder embedder \
    && mkdir -p /hf-cache /app \
    && chown embedder:embedder /hf-cache /app

WORKDIR /app
COPY embedder/pyproject.toml embedder/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    UV_CACHE_DIR=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY --chown=embedder:embedder embedder/app.py ./app.py

USER embedder
EXPOSE 8000
# 模型在背景執行緒載入；載入完成前 /health 回 503
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

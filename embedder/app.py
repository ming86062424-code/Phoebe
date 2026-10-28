"""Embedding service: sentence-transformers wrapper for text and images.

Model usage follows the Hugging Face card for ``google/embeddinggemma-2``:
``SentenceTransformer(id).encode(inputs, processing_kwargs={"image": {"max_soft_tokens": N}})``
where an input is a string, ``{"image": PIL.Image}`` or ``{"text": ..., "image": PIL.Image}``
(sentence-transformers 6.1: ``base/modality.py`` unwraps single-key dicts, ``base/model.py::preprocess``
forwards ``processing_kwargs`` to ``Transformer.preprocess``).
"""

import base64
import binascii
import io
import logging
import os
import threading
import warnings
from contextlib import asynccontextmanager
from typing import Annotated, Callable, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

log = logging.getLogger("embedder")

DEFAULT_MODEL_ID = "google/embeddinggemma-2"
DEFAULT_IMAGE_TOKENS = 280
MAX_ITEMS = 64
VALID_IMAGE_TOKENS = (70, 140, 280, 560, 1120)
# EMBED_DEVICE: "cuda" only with the GPU image (docker/docker-compose.gpu.yml); docs/gpu-memory.md
DEFAULT_DEVICE = "cpu"
VALID_DEVICES = ("cpu", "cuda")
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # decoded
MAX_IMAGE_B64_CHARS = (MAX_IMAGE_BYTES + 2) // 3 * 4
MAX_IMAGE_PIXELS = 25_000_000
MAX_BODY_BYTES = 100 * 1024 * 1024
QUERY_PREFIX = "task: search result | query: "


class Query(BaseModel):
    kind: Literal["query"]
    text: str


class Document(BaseModel):
    kind: Literal["document"]
    title: str
    text: str


class ImageItem(BaseModel):
    kind: Literal["image"]
    image_b64: str


class ImageText(BaseModel):
    kind: Literal["image_text"]
    image_b64: str
    text: str


Item = Annotated[Query | Document | ImageItem | ImageText, Field(discriminator="kind")]


class EmbedRequest(BaseModel):
    items: Annotated[list[Item], Field(max_length=MAX_ITEMS)]


def _bad_image(detail: str) -> HTTPException:
    return HTTPException(status_code=422, detail=detail)


def _decode_image(data: str) -> Image.Image:
    if len(data) > MAX_IMAGE_B64_CHARS:
        raise _bad_image("image too large")
    try:
        raw = base64.b64decode(data, validate=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(raw))
            if img.width * img.height > MAX_IMAGE_PIXELS:
                raise _bad_image("image has too many pixels")
            img.load()
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            background = Image.new("RGB", rgba.size, "white")
            background.paste(rgba, mask=rgba.getchannel("A"))
            return background
        return img.convert("RGB")
    except HTTPException:
        raise
    except (
        binascii.Error,
        ValueError,
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise _bad_image("invalid image_b64") from exc


def load_sentence_transformer(model_id: str, device: str = DEFAULT_DEVICE):
    # Imported lazily so tests (which inject a fake model) never import torch.
    import torch
    from sentence_transformers import SentenceTransformer

    if device == "cuda" and not torch.cuda.is_available():
        # never fall back to the CPU silently: the GPU mode was asked for
        raise RuntimeError(
            "EMBED_DEVICE=cuda but no CUDA GPU is visible "
            "(GPU image phoebe-embedder:gpu and docker-compose.gpu.yml are needed)"
        )
    # float32 on both devices: the model card forbids float16, and the RTX 2060 has no bfloat16.
    # audio_config=None: transformers never builds the audio encoder (304M parameters, ~1.2 GB
    # in float32) and skips its weights; only text and images are embedded, and their vectors
    # are identical to the full model's (2026-10-24: max difference 0.0).
    return SentenceTransformer(
        model_id,
        device=device,
        model_kwargs={"torch_dtype": torch.float32},
        config_kwargs={"audio_config": None},
    )


def free_gpu_cache() -> None:
    """Give the VRAM PyTorch keeps cached after a request back to other programs (Ollama)."""
    import torch

    torch.cuda.empty_cache()


def create_app(
    model=None,
    *,
    loader: Callable[[], object] | None = None,
    model_id: str | None = None,
    image_tokens: int | None = None,
    device: str | None = None,
    free_cache: Callable[[], None] | None = None,
    on_load_error: Callable[[], None] | None = None,
) -> FastAPI:
    """``model`` is injected (ready immediately); otherwise ``loader`` (default: the real model)
    runs in a background thread so the server starts answering /health with 503 right away."""
    model_id = model_id or os.environ.get("EMBED_MODEL_ID", DEFAULT_MODEL_ID)
    if image_tokens is None:
        image_tokens = int(os.environ.get("EMBED_IMAGE_TOKENS", DEFAULT_IMAGE_TOKENS))
    if image_tokens not in VALID_IMAGE_TOKENS:
        raise ValueError(
            f"EMBED_IMAGE_TOKENS must be one of {VALID_IMAGE_TOKENS}, got {image_tokens!r}"
        )
    device = device or os.environ.get("EMBED_DEVICE", DEFAULT_DEVICE)
    if device not in VALID_DEVICES:
        raise ValueError(f"EMBED_DEVICE must be one of {VALID_DEVICES}, got {device!r}")
    if free_cache is None and device == "cuda":
        free_cache = free_gpu_cache
    state = {"model": model, "error": None}
    lock = threading.Lock()  # one encode at a time; the model is not meant for concurrent calls

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["model"] is None:
            t = threading.Thread(target=load, name="model-loader", daemon=True)
            app.state.loading_thread = t
            t.start()
        yield

    app = FastAPI(title="Phoebe embedder", lifespan=lifespan)
    app.state.loading_thread = None

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "request body too large"}, status_code=413)
        return await call_next(request)

    def load():
        try:
            state["model"] = (loader or (lambda: load_sentence_transformer(model_id, device)))()
            log.info("model %s loaded on %s", model_id, device)
        except Exception as exc:  # keep serving 503 with the reason
            log.exception("model load failed")
            state["error"] = f"{type(exc).__name__}: {exc}"
            # exit so that `restart: unless-stopped` retries (e.g. an interrupted download)
            (on_load_error or (lambda: os._exit(1)))()

    @app.get("/health")
    def health():
        if state["model"] is None:
            detail = state["error"] or "model loading"
            raise HTTPException(status_code=503, detail=detail)
        return {"status": "ok", "model": model_id, "device": device}

    @app.post("/embed")
    def embed(req: EmbedRequest):
        m = state["model"]
        if m is None:
            raise HTTPException(status_code=503, detail=state["error"] or "model loading")
        if not req.items:
            return {"embeddings": []}

        text_idx, image_idx, mixed_idx = [], [], []
        text_in, image_in, mixed_in = [], [], []
        for i, item in enumerate(req.items):
            if item.kind == "query":
                text_idx.append(i)
                text_in.append(QUERY_PREFIX + item.text)
            elif item.kind == "document":
                text_idx.append(i)
                text_in.append(f"title: {item.title} | text: {item.text}")
            elif item.kind == "image":
                image_idx.append(i)
                image_in.append({"image": _decode_image(item.image_b64)})
            else:
                mixed_idx.append(i)
                mixed_in.append(
                    {"text": "<|image|> " + item.text, "image": _decode_image(item.image_b64)}
                )

        # a request is encoded in one forward pass per kind: the caller's batch size (ingest
        # sends 2 page images per request) bounds the memory peak
        image_kwargs = {"processing_kwargs": {"image": {"max_soft_tokens": image_tokens}}}
        out: np.ndarray | None = None
        with lock:
            for idx, inputs, kwargs in (
                (text_idx, text_in, {}),
                (image_idx, image_in, image_kwargs),
                (mixed_idx, mixed_in, image_kwargs),
            ):
                if inputs:
                    vecs = np.asarray(m.encode(inputs, **kwargs), dtype=np.float32)
                    if out is None:
                        out = np.zeros((len(req.items), vecs.shape[1]), dtype=np.float32)
                    out[idx] = vecs
            if free_cache is not None:
                free_cache()
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.maximum(norms, 1e-12)
        return {"embeddings": out.tolist()}

    return app


# `uvicorn app:app` — the real model is loaded in the background at startup.
app = create_app()

# embedder CPU／GPU 模式 Implementation Plan

**Goal:** embedder can run on CPU (default) or on the GPU, selected by an image variant plus a compose override file.

**Architecture:** `EMBED_DEVICE` (cpu|cuda) chooses the device at load time; `TORCH_VARIANT` (cpu|cu126) chooses the PyTorch wheel at build time; `docker/docker-compose.gpu.yml` switches the embedder service to the GPU variant.

**Tech Stack:** sentence-transformers 6.1, torch 2.14 (cpu / cu126 wheels), uv dependency groups with conflicts, Docker Compose override files.

**Spec:** docs/superpowers/specs/2026-10-24-embedder-gpu-mode-design.md

## Global Constraints

- float32 on both devices; never float16 or bfloat16.
- `EMBED_DEVICE` accepts exactly `cpu` and `cuda`; default `cpu`.
- CPU image stays `phoebe-embedder:local`; GPU image is `phoebe-embedder:gpu`.
- The audio encoder stays skipped (`config_kwargs={"audio_config": None}`).

## Review Focus

- `EMBED_DEVICE=cuda` with no GPU visible: load fails with a clear log line and the process exits (restart loop), never silently on CPU.
- `uv sync` with no group flags (local tests) still installs the CPU wheel.
- A plain `docker compose up -d` after GPU mode recreates the embedder on CPU (documented, expected).
- GPU mode together with `--profile lightonocr`: documented as unsupported.

---

### Task 1: Device setting in the embedder

**Files:** Modify `embedder/app.py`; Test `embedder/tests/test_app.py`

- `VALID_DEVICES = ("cpu", "cuda")`; `create_app(..., device: str | None = None)` reads `EMBED_DEVICE` (default `cpu`) and raises `ValueError` for other values.
- `load_sentence_transformer(model_id, device="cpu")`: when `device == "cuda"` and `torch.cuda.is_available()` is false, raise `RuntimeError("EMBED_DEVICE=cuda but no CUDA GPU is visible ...")`.
- The default loader passes the device; `/health` returns `{"status": "ok", "model": ..., "device": ...}`.
- Tests: invalid device rejected; loader passes `device="cuda"`; cuda without GPU raises; health reports the device.

### Task 2: Two PyTorch variants

**Files:** Modify `embedder/pyproject.toml`, `embedder/uv.lock`, `docker/embedder.Dockerfile`

- dependency groups `cpu` and `cu126`, each `torch==2.14.0`, `torchvision==0.29.0`; `[tool.uv] conflicts` between them; `default-groups = ["dev", "cpu"]`; sources map torch/torchvision to index `pytorch-cpu` (group cpu) or `pytorch-cu126` (group cu126); torch/torchvision leave the main dependencies.
- Dockerfile: `ARG TORCH_VARIANT=cpu`; `uv sync --frozen --no-default-groups --group ${TORCH_VARIANT} --no-install-project`.
- Verify: `uv lock`, local `uv sync` installs `+cpu`, both images build.

### Task 3: GPU compose override

**Files:** Create `docker/docker-compose.gpu.yml`; Test `tests/test_config.py`

- embedder: `build.args.TORCH_VARIANT: cu126`, `image: phoebe-embedder:gpu`, `environment.EMBED_DEVICE: cuda`, GPU device reservation (same as vllm).
- Base compose embedder sets `EMBED_DEVICE: cpu` explicitly.
- Tests assert those values and that the Dockerfile has the build arg.

### Task 4: Live verification and docs

- Build GPU image, start GPU mode (vllm stopped), check health/device, VRAM and RAM, CPU-vs-GPU vector cosine, LiteParse indexing time on the strawberry PDF; switch back to LightOnOCR mode.
- Docs: `docs/gpu-memory.md` (embedder section, mode switching), `docker/README.md`, `docs/services.md`.

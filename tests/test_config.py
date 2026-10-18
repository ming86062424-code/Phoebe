import inspect
import json
import os
import re
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, Settings


def _env_file(tmp_path, monkeypatch, text):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setenv("ENV_FILE", str(path))


def test_env_file_values_are_loaded(tmp_path, monkeypatch):
    _env_file(tmp_path, monkeypatch, "CHAT_MODEL=from-file\n")
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    try:
        assert Settings.from_env().chat_model == "from-file"
    finally:
        os.environ.pop("CHAT_MODEL", None)


def test_environment_overrides_env_file(tmp_path, monkeypatch):
    _env_file(tmp_path, monkeypatch, "CHAT_MODEL=from-file\n")
    monkeypatch.setenv("CHAT_MODEL", "from-env")
    assert Settings.from_env().chat_model == "from-env"


def test_tests_ignore_developer_env_file():
    path = Path(os.environ["ENV_FILE"])
    assert not path.exists()
    assert path != PROJECT_ROOT / ".env"


def test_new_settings_defaults(monkeypatch):
    for key in (
        "KB_WORKER_MODE",
        "OCR_BACKEND",
        "VLLM_BASE_URL",
        "CHROMA_HOST",
        "CHROMA_PORT",
        "REPLY_LOCK_TTL_SECONDS",
        "KB_WORKER_POLL_SECONDS",
        "KB_STALE_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    s = Settings.from_env()
    assert s.kb_worker_mode == "embedded"
    assert s.ocr_backend == "transformers"
    assert s.vllm_base_url == "http://localhost:8001/v1"
    assert s.chroma_host == ""
    assert s.chroma_port == 8000
    assert s.reply_lock_ttl_seconds == 90
    assert s.kb_worker_poll_seconds == 2.0
    assert s.kb_stale_seconds == 120
    assert s.kb_heartbeat_seconds == 30


def test_invalid_worker_mode_and_ocr_backend_raise(monkeypatch):
    monkeypatch.setenv("KB_WORKER_MODE", "nope")
    with pytest.raises(ValueError, match="KB_WORKER_MODE"):
        Settings.from_env()
    monkeypatch.setenv("KB_WORKER_MODE", "external")
    monkeypatch.setenv("OCR_BACKEND", "nope")
    with pytest.raises(ValueError, match="OCR_BACKEND"):
        Settings.from_env()
    monkeypatch.setenv("OCR_BACKEND", "vllm")
    s = Settings.from_env()
    assert (s.kb_worker_mode, s.ocr_backend) == ("external", "vllm")


def test_env_example_lists_every_setting():
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    keys = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", text, flags=re.M))
    source = inspect.getsource(Settings.from_env)
    used = set(re.findall(r'(?:os\.environ\.get|_choice)\(\s*"([A-Z0-9_]+)"', source))
    assert {"KB_WORKER_MODE", "OCR_BACKEND"} <= used
    assert used
    assert used <= keys
    assert not {"WEB_CONCURRENCY", "PHOEBE_PORT", "VLLM_GPU_UTIL"} & keys


def test_docker_env_example_lists_compose_and_settings_keys():
    import yaml

    compose_text = (PROJECT_ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8")
    compose = yaml.safe_load(compose_text)
    interpolated = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)", compose_text))
    fixed = set()
    for name in ("web", "worker"):
        fixed |= set(compose["services"][name]["environment"])
    fixed |= {"FRONTEND_DIST", "CHROMA_DIR"}  # not configurable in containers

    text = (PROJECT_ROOT / "docker" / ".env.example").read_text(encoding="utf-8")
    keys = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", text, flags=re.M))
    source = inspect.getsource(Settings.from_env)
    used = set(re.findall(r'(?:os\.environ\.get|_choice)\(\s*"([A-Z0-9_]+)"', source))

    assert interpolated and used
    assert interpolated <= keys
    assert (used - fixed) <= keys
    assert not (fixed & keys)
    assert "WEB_CONCURRENCY" in keys


def test_heartbeat_must_be_shorter_than_stale_timeout(monkeypatch):
    monkeypatch.setenv("KB_HEARTBEAT_SECONDS", "120")
    monkeypatch.setenv("KB_STALE_SECONDS", "120")
    with pytest.raises(ValueError, match="KB_HEARTBEAT_SECONDS"):
        Settings.from_env()
    monkeypatch.setenv("KB_HEARTBEAT_SECONDS", "119")
    assert Settings.from_env().kb_heartbeat_seconds == 119


def test_reply_lock_ttl_must_exceed_heartbeat(monkeypatch):
    from app.config import REPLY_HEARTBEAT_SECONDS

    monkeypatch.setenv("REPLY_LOCK_TTL_SECONDS", str(REPLY_HEARTBEAT_SECONDS))
    with pytest.raises(ValueError, match="REPLY_LOCK_TTL_SECONDS"):
        Settings.from_env()
    monkeypatch.setenv("REPLY_LOCK_TTL_SECONDS", str(REPLY_HEARTBEAT_SECONDS + 1))
    assert Settings.from_env().reply_lock_ttl_seconds == REPLY_HEARTBEAT_SECONDS + 1


def test_vllm_service_follows_official_flags():
    import yaml

    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8")
    )
    vllm = compose["services"]["vllm"]
    assert "profiles" not in vllm  # a default service
    command = vllm["command"]
    assert vllm["image"] == "vllm/vllm-openai:v0.30.0"
    for flag in ("--trust-remote-code", "--no-enable-prefix-caching", "float16"):
        assert flag in command
    flags = {command[i + 1] for i, c in enumerate(command) if c.startswith("--") and i + 1 < len(command)}
    kwargs = command[command.index("--default-chat-template-kwargs") + 1]
    assert json.loads(kwargs) == {"enable_thinking": False}
    assert json.loads(command[command.index("--limit-mm-per-prompt") + 1]) == {"image": 1}
    assert command[command.index("--mm-processor-cache-gb") + 1] == "0"
    assert "${VLLM_MAX_MODEL_LEN:-18000}" in command
    assert "${VLLM_GPU_UTIL:-0.85}" in command
    assert flags
    assert "healthcheck" in vllm
    assert "hf-cache:/root/.cache/huggingface" in vllm["volumes"]


def test_worker_uses_vllm_and_has_no_gpu():
    import yaml

    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8")
    )
    worker = compose["services"]["worker"]
    assert worker["environment"]["OCR_BACKEND"] == "vllm"
    assert worker["environment"]["VLLM_BASE_URL"] == "http://vllm:8000/v1"
    assert worker["depends_on"]["vllm"]["condition"] == "service_healthy"
    assert "deploy" not in worker
    assert "HF_HOME" not in worker["environment"]
    assert not any("hf-cache" in v for v in worker.get("volumes", []))


def test_dockerignore_is_an_allowlist_covering_dockerfile_copies():
    lines = [
        ln.strip()
        for ln in (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert lines[0] == "*"
    allowed = {ln[1:].rstrip("/") for ln in lines if ln.startswith("!")}
    dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    sources = []
    for ln in dockerfile.splitlines():
        m = re.match(r"COPY\s+((?:--\S+\s+)*)(.+)$", ln)
        if not m or "--from=" in m.group(1):
            continue
        *srcs, _dest = m.group(2).split()
        sources += [s.rstrip("/") for s in srcs]
    assert sources
    for src in sources:
        # a source is reachable if it, or one of its parent folders, is allowed
        parts = src.split("/")
        assert any("/".join(parts[: i + 1]) in allowed for i in range(len(parts))), src

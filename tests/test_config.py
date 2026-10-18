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
    command = vllm["command"]
    assert "--trust-remote-code" in command
    i = command.index("--default-chat-template-kwargs")
    assert json.loads(command[i + 1]) == {"enable_thinking": False}
    assert vllm["image"] == "vllm/vllm-openai:v0.30.0"
    assert "float16" in command

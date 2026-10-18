import inspect
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
    assert s.reply_lock_ttl_seconds == 600
    assert s.kb_worker_poll_seconds == 2.0
    assert s.kb_stale_seconds == 900


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
    used = set(re.findall(r'os\.environ\.get\(\s*"([A-Z0-9_]+)"', inspect.getsource(Settings.from_env)))
    assert used
    assert used <= keys
    assert {"WEB_CONCURRENCY", "PHOEBE_PORT", "VLLM_GPU_UTIL"} <= keys

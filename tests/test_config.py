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
        "KB_INGEST_MODE",
        "EMBEDDER_URL",
        "OCR_BACKEND",
        "VLLM_BASE_URL",
        "MILVUS_URI",
        "MILVUS_TOKEN",
        "MILVUS_COLLECTION",
        "REPLY_LOCK_TTL_SECONDS",
        "KB_INGEST_POLL_SECONDS",
        "KB_STALE_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    s = Settings.from_env()
    assert s.kb_ingest_mode == "embedded"
    assert s.embedder_url == "http://localhost:8002"
    assert s.ocr_backend == "transformers"
    assert s.vllm_base_url == "http://localhost:8001/v1"
    assert s.milvus_uri == "http://localhost:19530"
    assert s.milvus_token == ""
    assert s.milvus_collection == "phoebe_chunks"
    assert s.reply_lock_ttl_seconds == 90
    assert s.kb_ingest_poll_seconds == 2.0
    assert s.kb_stale_seconds == 120
    assert s.kb_heartbeat_seconds == 30


def test_invalid_ingest_mode_and_ocr_backend_raise(monkeypatch):
    monkeypatch.setenv("KB_INGEST_MODE", "nope")
    with pytest.raises(ValueError, match="KB_INGEST_MODE"):
        Settings.from_env()
    monkeypatch.setenv("KB_INGEST_MODE", "external")
    monkeypatch.setenv("OCR_BACKEND", "nope")
    with pytest.raises(ValueError, match="OCR_BACKEND"):
        Settings.from_env()
    monkeypatch.setenv("OCR_BACKEND", "vllm")
    s = Settings.from_env()
    assert (s.kb_ingest_mode, s.ocr_backend) == ("external", "vllm")


# Old settings that Settings.from_env still reads, only to fail with a hint
RETIRED_KEYS = {"ADMIN_USERNAMES", "KB_WORKER_MODE", "KB_WORKER_POLL_SECONDS"}


def test_env_example_lists_every_setting():
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    keys = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", text, flags=re.M))
    source = inspect.getsource(Settings.from_env)
    used = set(re.findall(r'(?:os\.environ\.get|_choice)\(\s*"([A-Z0-9_]+)"', source))
    used -= RETIRED_KEYS  # read only to reject them with a clear message
    assert {"KB_INGEST_MODE", "OCR_BACKEND"} <= used
    assert used
    assert used <= keys
    assert not {"WEB_CONCURRENCY", "PHOEBE_PORT", "VLLM_GPU_UTIL", "OCR_VLLM_GPU_UTIL"} & keys
    assert not RETIRED_KEYS & keys


def test_docker_env_example_lists_compose_and_settings_keys():
    import yaml

    compose_text = (PROJECT_ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8")
    compose = yaml.safe_load(compose_text)
    interpolated = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)", compose_text))
    fixed = set()
    for name in ("web", "ingest"):
        fixed |= set(compose["services"][name]["environment"])
    fixed |= {"FRONTEND_DIST"}  # not configurable in containers

    text = (PROJECT_ROOT / "docker" / ".env.example").read_text(encoding="utf-8")
    keys = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", text, flags=re.M))
    source = inspect.getsource(Settings.from_env)
    used = set(re.findall(r'(?:os\.environ\.get|_choice)\(\s*"([A-Z0-9_]+)"', source))
    used -= RETIRED_KEYS  # read only to reject them with a clear message

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
    assert vllm["profiles"] == ["lightonocr"]  # opt-in service
    command = vllm["command"]
    assert vllm["image"] == "vllm/vllm-openai:v0.30.0"
    for flag in ("--trust-remote-code", "--no-enable-prefix-caching", "float16"):
        assert flag in command
    kwargs = command[command.index("--default-chat-template-kwargs") + 1]
    assert json.loads(kwargs) == {"enable_thinking": False}
    assert json.loads(command[command.index("--limit-mm-per-prompt") + 1]) == {"image": 1}
    assert command[command.index("--mm-processor-cache-gb") + 1] == "0"
    assert "${OCR_VLLM_MAX_MODEL_LEN:-18000}" in command
    assert "${OCR_VLLM_GPU_UTIL:-0.8}" in command
    assert "--enforce-eager" in command
    assert command[command.index("--max-num-seqs") + 1] == "${OCR_VLLM_MAX_NUM_SEQS:-4}"
    assert "env_file" not in vllm  # no app settings, so no "unknown VLLM_*" warnings
    health = vllm["healthcheck"]
    assert any("/health" in str(part) for part in health["test"])
    assert health["start_period"]
    assert vllm["ports"] == ["127.0.0.1:8001:8000"]
    assert "./volumes/vllm-hf-cache:/root/.cache/huggingface" in vllm["volumes"]


def test_ingest_uses_vllm_and_has_no_gpu():
    import yaml

    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8")
    )
    ingest = compose["services"]["ingest"]
    assert ingest["environment"]["OCR_BACKEND"] == "vllm"
    assert ingest["environment"]["VLLM_BASE_URL"] == "http://vllm:8000/v1"
    # LiteParse must keep working while vLLM is down or still downloading
    assert ingest["depends_on"]["vllm"]["condition"] == "service_started"
    assert "deploy" not in ingest
    assert "HF_HOME" not in ingest["environment"]
    assert not any("hf-cache" in v for v in ingest.get("volumes", []))


def _compose():
    import yaml

    return yaml.safe_load(
        (PROJECT_ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8")
    )


def test_milvus_stack_in_compose():
    services = _compose()["services"]
    for name in ("milvus", "etcd", "minio"):
        svc = services[name]
        assert "profiles" not in svc
        assert svc["healthcheck"]["test"]
        assert re.search(r":v?\d+\.\d+", svc["image"]) or "RELEASE." in svc["image"]
        assert "container_name" not in svc
    milvus = services["milvus"]
    assert milvus["image"].startswith("milvusdb/milvus:v")
    assert milvus["ports"] == ["127.0.0.1:19530:19530"]
    assert set(milvus["depends_on"]) == {"etcd", "minio"}
    for dep in milvus["depends_on"].values():
        assert dep["condition"] == "service_healthy"
    assert any("9091/healthz" in str(p) for p in milvus["healthcheck"]["test"])


def test_web_and_ingest_wait_for_healthy_milvus():
    services = _compose()["services"]
    for name in ("web", "ingest"):
        svc = services[name]
        assert svc["depends_on"]["milvus"]["condition"] == "service_healthy"
        assert svc["environment"]["MILVUS_URI"] == "http://milvus:19530"
        assert not any(k.startswith("CHROMA_") for k in svc["environment"])


def test_no_chroma_left_in_compose():
    compose = _compose()
    assert "chroma" not in compose["services"]
    assert "chroma-data" not in compose["volumes"]


def test_service_data_lives_under_docker_volumes_except_sqlite():
    compose = _compose()
    services = compose["services"]
    expected = {
        "etcd": "./volumes/etcd:/etcd",
        "minio": "./volumes/minio:/minio_data",
        "milvus": "./volumes/milvus:/var/lib/milvus",
        "embedder": "./volumes/embedder-hf-cache:/hf-cache",
        "vllm": "./volumes/vllm-hf-cache:/root/.cache/huggingface",
    }
    for name, mount in expected.items():
        assert services[name]["volumes"] == [mount]
    # SQLite (WAL, shared by two containers) stays on a Docker-managed volume
    assert list(compose["volumes"]) == ["phoebe-db"]
    for name in ("web", "ingest"):
        svc = services[name]
        assert svc["volumes"] == ["phoebe-db:/db", "./volumes/kb:/data/kb"]
        assert svc["environment"]["DB_PATH"] == "/db/app.db"
        assert svc["environment"]["KB_DIR"] == "/data/kb"


def test_docker_volumes_dir_is_git_ignored():
    lines = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "docker/volumes/" in lines


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


def test_gemini_api_key_not_in_repr():
    from app.config import Settings

    assert "sk-secret" not in repr(Settings(gemini_api_key="sk-secret"))


def test_dockerfile_has_no_literal_backslash_n_outside_comments():
    from pathlib import Path

    text = (Path(__file__).parent.parent / "docker" / "Dockerfile").read_text(encoding="utf-8")
    bad = [
        line
        for line in text.splitlines()
        if not line.lstrip().startswith("#") and chr(92) + "n" in line
    ]
    assert bad == []


def test_kb_max_images_defaults_to_four_and_must_be_a_non_negative_int(monkeypatch):
    monkeypatch.delenv("KB_MAX_IMAGES", raising=False)
    assert Settings.from_env().kb_max_images == 4
    monkeypatch.setenv("KB_MAX_IMAGES", "0")
    assert Settings.from_env().kb_max_images == 0
    for bad in ("-1", "two", "1.5"):
        monkeypatch.setenv("KB_MAX_IMAGES", bad)
        with pytest.raises(ValueError, match="KB_MAX_IMAGES"):
            Settings.from_env()


@pytest.mark.parametrize(("old", "new"), [
    ("KB_WORKER_MODE", "KB_INGEST_MODE"),
    ("KB_WORKER_POLL_SECONDS", "KB_INGEST_POLL_SECONDS"),
])
def test_old_worker_settings_are_rejected(monkeypatch, old, new):
    monkeypatch.setenv(old, "2")
    with pytest.raises(ValueError, match=f"{old} 已改名為 {new}"):
        Settings.from_env()


def test_ingest_settings(monkeypatch):
    monkeypatch.setenv("KB_INGEST_MODE", "external")
    monkeypatch.setenv("KB_INGEST_POLL_SECONDS", "3")
    s = Settings.from_env()
    assert (s.kb_ingest_mode, s.kb_ingest_poll_seconds) == ("external", 3.0)


def test_compose_runs_the_ingest_service():
    services = _compose()["services"]
    assert "worker" not in services
    assert services["ingest"]["build"]["target"] == "ingest"
    assert services["ingest"]["image"] == "phoebe-ingest:local"
    # in-flight OCR pages (4 at once, ~90 s) must finish before Docker SIGKILLs ingest
    assert services["ingest"]["stop_grace_period"] == "120s"


def test_vllm_is_opt_in_behind_the_lightonocr_profile():
    services = _compose()["services"]
    assert services["vllm"]["profiles"] == ["lightonocr"]
    dep = services["ingest"]["depends_on"]["vllm"]
    assert dep == {"condition": "service_started", "required": False}
    command = services["vllm"]["command"]
    assert command[command.index("--max-num-seqs") + 1] == "${OCR_VLLM_MAX_NUM_SEQS:-4}"


def test_docker_env_example_defaults_to_four_sequences():
    text = (PROJECT_ROOT / "docker" / ".env.example").read_text(encoding="utf-8")
    assert "OCR_VLLM_MAX_NUM_SEQS=4" in text


def test_ingest_concurrency_defaults_to_one_and_must_be_positive(monkeypatch):
    monkeypatch.delenv("INGEST_CONCURRENCY", raising=False)
    assert Settings.from_env().ingest_concurrency == 1
    monkeypatch.setenv("INGEST_CONCURRENCY", "3")
    assert Settings.from_env().ingest_concurrency == 3
    for bad in ("0", "abc", "-1"):
        monkeypatch.setenv("INGEST_CONCURRENCY", bad)
        with pytest.raises(ValueError, match="INGEST_CONCURRENCY 必須是正整數"):
            Settings.from_env()


def test_env_examples_document_ingest_concurrency():
    for path in (PROJECT_ROOT / ".env.example", PROJECT_ROOT / "docker" / ".env.example"):
        text = path.read_text(encoding="utf-8")
        assert "INGEST_CONCURRENCY=1" in text and "同時處理幾份文件" in text


def test_ocr_concurrency_must_be_positive(monkeypatch):
    monkeypatch.delenv("OCR_CONCURRENCY", raising=False)
    assert Settings.from_env().ocr_concurrency == 4
    monkeypatch.setenv("OCR_CONCURRENCY", "2")
    assert Settings.from_env().ocr_concurrency == 2
    for bad in ("0", "abc", "-1"):
        monkeypatch.setenv("OCR_CONCURRENCY", bad)
        with pytest.raises(ValueError, match="OCR_CONCURRENCY 必須是正整數"):
            Settings.from_env()


def test_ocr_timeout_defaults_to_600_and_must_be_positive(monkeypatch):
    monkeypatch.delenv("OCR_TIMEOUT_SECONDS", raising=False)
    assert Settings.from_env().ocr_timeout_seconds == 600
    monkeypatch.setenv("OCR_TIMEOUT_SECONDS", "90.5")
    assert Settings.from_env().ocr_timeout_seconds == 90.5
    for bad in ("0", "abc", "-1", "nan", "inf"):
        monkeypatch.setenv("OCR_TIMEOUT_SECONDS", bad)
        with pytest.raises(ValueError, match="OCR_TIMEOUT_SECONDS 必須是正數"):
            Settings.from_env()


def test_env_examples_document_ocr_concurrency():
    for path in (PROJECT_ROOT / ".env.example", PROJECT_ROOT / "docker" / ".env.example"):
        text = path.read_text(encoding="utf-8")
        assert "OCR_CONCURRENCY=4" in text and "OCR_TIMEOUT_SECONDS=600" in text
        assert "每份文件同時送給 vLLM 的頁數" in text
        assert "--max-num-seqs" in text


def test_kb_services_pass_the_ocr_concurrency():
    source = (PROJECT_ROOT / "app" / "kb" / "services.py").read_text(encoding="utf-8")
    assert "LightOnOcrParser(build_ocr_engine(settings), settings.ocr_concurrency)" in source

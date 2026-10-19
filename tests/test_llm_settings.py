import pytest

from app.config import Settings
from app.db import connect, init_db
from app.llm_settings import load_llm_settings, save_llm_settings


@pytest.fixture
def conn(tmp_path):
    c = connect(str(tmp_path / "s.db"))
    init_db(c)
    yield c
    c.close()


@pytest.fixture
def env_settings():
    return Settings(chat_model="chat-x", gemini_model="gem-x", llm_provider="ollama")


def test_defaults_come_from_env(conn, env_settings):
    s = load_llm_settings(conn, env_settings)
    assert s.llm_provider == "ollama"
    assert s.ollama_model == "chat-x"
    assert s.gemini_model == "gem-x"
    assert s.system_prompt == ""
    assert s.temperature is None


def test_partial_update_keeps_other_keys(conn, env_settings):
    save_llm_settings(conn, env_settings, {"system_prompt": "hi", "temperature": 0.5})
    s = save_llm_settings(conn, env_settings, {"ollama_model": "m2"})
    assert (s.system_prompt, s.temperature, s.ollama_model) == ("hi", 0.5, "m2")
    assert s.gemini_model == "gem-x"
    assert load_llm_settings(conn, env_settings) == s


def test_temperature_can_be_reset_to_null(conn, env_settings):
    save_llm_settings(conn, env_settings, {"temperature": 1})
    s = save_llm_settings(conn, env_settings, {"temperature": None})
    assert s.temperature is None


@pytest.mark.parametrize(
    "values",
    [
        {"llm_provider": "x"},
        {"temperature": 2.5},
        {"temperature": -0.1},
        {"system_prompt": "a" * 4001},
        {"ollama_model": "  "},
        {"gemini_model": ""},
        {"nope": 1},
    ],
)
def test_validation_rejects_bad_values(conn, env_settings, values):
    with pytest.raises(ValueError):
        save_llm_settings(conn, env_settings, values)
    assert load_llm_settings(conn, env_settings).llm_provider == "ollama"


def test_boundaries_are_accepted(conn, env_settings):
    s = save_llm_settings(
        conn, env_settings, {"temperature": 2, "system_prompt": "a" * 4000}
    )
    assert s.temperature == 2.0 and len(s.system_prompt) == 4000


def test_gemini_env_settings(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    s = Settings.from_env()
    assert (s.gemini_api_key, s.gemini_model) == ("", "gemini-2.5-flash")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert Settings.from_env().gemini_api_key == "k"


def test_default_provider_follows_settings(conn):
    assert load_llm_settings(conn, Settings(llm_provider="gemini")).llm_provider == "gemini"
    assert load_llm_settings(conn, Settings(llm_provider="ollama")).llm_provider == "ollama"


def test_llm_provider_and_ollama_context_env(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    s = Settings.from_env()
    assert (s.llm_provider, s.ollama_num_ctx) == ("gemini", 16384)
    monkeypatch.setenv("LLM_PROVIDER", " Ollama ")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    s = Settings.from_env()
    assert (s.llm_provider, s.ollama_num_ctx) == ("ollama", 8192)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("LLM_PROVIDER", "openai", "LLM_PROVIDER 必須是 ollama 或 gemini"),
        ("OLLAMA_NUM_CTX", "0", "OLLAMA_NUM_CTX 必須是正整數"),
        ("OLLAMA_NUM_CTX", "lots", "OLLAMA_NUM_CTX 必須是正整數"),
    ],
)
def test_invalid_provider_or_context_env(monkeypatch, name, value, message):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        Settings.from_env()

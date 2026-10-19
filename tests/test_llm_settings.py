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
    return Settings(chat_model="chat-x", gemini_model="gem-x")


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


def test_admin_usernames_parsing(monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAMES", " alice, bob ,,")
    assert Settings.from_env().admin_usernames == ("alice", "bob")
    monkeypatch.delenv("ADMIN_USERNAMES")
    assert Settings.from_env().admin_usernames == ()


def test_gemini_env_settings(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    s = Settings.from_env()
    assert (s.gemini_api_key, s.gemini_model) == ("", "gemini-2.5-flash")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert Settings.from_env().gemini_api_key == "k"

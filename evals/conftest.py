import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

import asyncio  # noqa: E402
from pathlib import Path  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402
from dotenv import dotenv_values  # noqa: E402

from app.config import Settings  # noqa: E402
from app.ollama_client import OllamaClient  # noqa: E402

DOCKER_ENV = Path(__file__).resolve().parent.parent / "docker" / ".env"


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings.from_env()


@pytest.fixture(scope="session")
def config_value(settings: Settings):
    """A setting from the environment or the root .env (loaded by Settings.from_env), else
    from docker/.env, where the Docker deployment keeps GEMINI_API_KEY and the admin."""
    docker = {k: v for k, v in dotenv_values(DOCKER_ENV).items() if v} if DOCKER_ENV.exists() else {}

    def get(key: str, default: str | None = None) -> str | None:
        return os.environ.get(key) or docker.get(key) or default

    return get


def _ollama_models(settings: Settings) -> set[str]:
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        return {m.get("name") for m in resp.json().get("models", [])}
    except Exception as exc:
        pytest.skip(f"無法連線到 Ollama（{settings.ollama_base_url}）：{exc}")


def _make_judge_class():
    from deepeval.models import OllamaModel

    class NoThinkOllamaModel(OllamaModel):
        """qwen3.5 是 thinking 模型：思考過程會耗盡輸出，導致結構化 JSON 為空字串。
        DeepEval 的 OllamaModel 不支援 think 參數，這裡在 client 層強制 think=False。"""

        def _build_client(self, cls):
            class NoThinkClient(cls):
                def chat(self, *args, **kwargs):
                    kwargs.setdefault("think", False)
                    return super().chat(*args, **kwargs)

            return super()._build_client(NoThinkClient)

    return NoThinkOllamaModel


@pytest.fixture(scope="session")
def judge(settings: Settings, config_value):
    """EVAL_JUDGE_MODEL: a name starting with gemini- uses Gemini (GEMINI_API_KEY),
    anything else a model on the local Ollama."""
    name = config_value("EVAL_JUDGE_MODEL", settings.eval_judge_model)
    if name.startswith("gemini-"):
        from deepeval.models import GeminiModel

        key = config_value("GEMINI_API_KEY")
        if not key:
            pytest.skip("裁判模型是 Gemini，但沒有設定 GEMINI_API_KEY")
        return GeminiModel(model=name, api_key=key, temperature=0)
    if name not in _ollama_models(settings):
        pytest.skip(f"Ollama 缺少裁判模型：{name}")
    return _make_judge_class()(model=name, base_url=settings.ollama_base_url, temperature=0)


@pytest.fixture(scope="session")
def generate_reply(settings: Settings):
    """A reply straight from the local Ollama chat model (CHAT_MODEL), no knowledge base."""
    if settings.chat_model not in _ollama_models(settings):
        pytest.skip(f"Ollama 缺少聊天模型：{settings.chat_model}")

    def _generate(messages: list[dict]) -> str:
        client = OllamaClient(
            settings.ollama_base_url, settings.chat_model, think=settings.chat_think
        )

        async def collect() -> str:
            return "".join(
                [d.text async for d in client.stream_chat(messages) if d.kind == "content"]
            )

        return asyncio.run(collect())

    return _generate

import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

import asyncio  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402

from app.config import Settings  # noqa: E402
from app.ollama_client import OllamaClient  # noqa: E402


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings.from_env()


@pytest.fixture(scope="session", autouse=True)
def require_ollama(settings: Settings) -> None:
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        names = {m.get("name") for m in resp.json().get("models", [])}
    except Exception as exc:
        pytest.skip(f"無法連線到 Ollama（{settings.ollama_base_url}）：{exc}")
    missing = [
        m for m in (settings.chat_model, settings.eval_judge_model) if m not in names
    ]
    if missing:
        pytest.skip(f"Ollama 缺少模型：{', '.join(missing)}")


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
def judge(settings: Settings):
    return _make_judge_class()(
        model=settings.eval_judge_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )


@pytest.fixture(scope="session")
def generate_reply(settings: Settings):
    def _generate(messages: list[dict]) -> str:
        client = OllamaClient(settings.ollama_base_url, settings.chat_model)

        async def collect() -> str:
            return "".join(
                [d.text async for d in client.stream_chat(messages) if d.kind == "content"]
            )

        return asyncio.run(collect())

    return _generate

from app.config import Settings
from app.llm.base import ChatClient
from app.llm.gemini import GeminiChatClient
from app.llm.ollama import OllamaChatClient
from app.llm_settings import LlmSettings

PROVIDER_LABELS = {"ollama": "Ollama", "gemini": "Gemini"}


def build_chat_client(settings: Settings, llm: LlmSettings) -> ChatClient:
    if llm.llm_provider == "gemini":
        return GeminiChatClient(
            settings.gemini_api_key, llm.gemini_model, think=settings.chat_think
        )
    return OllamaChatClient(
        settings.ollama_base_url,
        llm.ollama_model,
        think=settings.chat_think,
        num_ctx=settings.ollama_num_ctx,
    )


async def provider_availability(settings: Settings) -> dict[str, tuple[bool, str]]:
    ollama = OllamaChatClient(
        settings.ollama_base_url, settings.chat_model, think=settings.chat_think
    )
    gemini = GeminiChatClient(
        settings.gemini_api_key, settings.gemini_model, think=settings.chat_think
    )
    return {
        "ollama": await ollama.availability(),
        "gemini": await gemini.availability(),
    }

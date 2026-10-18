from app.config import Settings
from app.llm.ollama import OllamaChatClient

PROVIDER_LABELS = {"ollama": "Ollama", "gemini": "Gemini"}


async def provider_availability(settings: Settings) -> dict[str, tuple[bool, str]]:
    ollama = OllamaChatClient(
        settings.ollama_base_url, settings.chat_model, think=settings.chat_think
    )
    has_key = bool(settings.gemini_api_key)
    return {
        "ollama": await ollama.availability(),
        "gemini": (has_key, "" if has_key else "未設定 GEMINI_API_KEY"),
    }

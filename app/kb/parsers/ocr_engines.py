import base64
import importlib.util
import io
import logging
from typing import Protocol

import httpx

from app.config import Settings
from app.kb.parsers.base import ParseError

logger = logging.getLogger(__name__)

# the model card uses 1024 in its transformers example and 4096 for vLLM; use 4096 so
# dense pages aren't truncated
MAX_NEW_TOKENS = 4096
# sampling settings from the official model card
TEMPERATURE = 0.2
TOP_P = 0.9
VLLM_UNAVAILABLE = "OCR 服務（vLLM）未啟動"


class OcrEngine(Protocol):
    def availability(self) -> tuple[bool, str]: ...

    def ocr(self, image) -> str:
        """OCR one PIL image and return its text."""
        ...

    def release(self) -> None: ...


def unload_ollama_models(base_url: str) -> None:
    """Ask Ollama to drop resident models so the OCR model gets the GPU memory."""
    base = base_url.rstrip("/")
    try:
        with httpx.Client(timeout=10) as client:
            running = client.get(f"{base}/api/ps").json().get("models", [])
            for m in running:
                name = m.get("name") or m.get("model")
                client.post(f"{base}/api/generate", json={"model": name, "keep_alive": 0})
    except httpx.HTTPError as e:
        logger.warning("無法請 Ollama 卸載模型：%s", e)


class TransformersOcrEngine:
    def __init__(self, model_id: str, ollama_base_url: str):
        self.model_id = model_id
        self.ollama_base_url = ollama_base_url
        self._model = None
        self._processor = None

    def availability(self) -> tuple[bool, str]:
        missing = [m for m in ("torch", "transformers") if importlib.util.find_spec(m) is None]
        if missing:
            return False, "未安裝 OCR 套件，請以 uv run --group ocr 啟動伺服器"
        return True, ""

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        unload_ollama_models(self.ollama_base_url)
        if torch.cuda.is_available():
            device = "cuda"
            # The model card recommends bf16; fp16 is only a hardware fallback, because
            # Turing and older GPUs (e.g. RTX 2060) lack native bf16.
            major, _ = torch.cuda.get_device_capability()
            dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            device, dtype = "cpu", torch.float32
        self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_id, dtype=dtype
        ).to(device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def ocr(self, image) -> str:
        import torch

        try:
            self._ensure_loaded()
        except Exception as e:
            self._model = None
            self._processor = None
            raise ParseError(f"LightOnOCR 初始化失敗：{e}") from e
        conversation = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        inputs = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self._model.device)
        # Official sampling settings (temperature 0.2), so re-running OCR may differ slightly.
        # The model config ships use_cache=false; without the KV cache every new
        # token recomputes the whole ~3k-token image prompt (~18x slower).
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                use_cache=True,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        return self._processor.decode(new_tokens, skip_special_tokens=True).strip()

    def release(self) -> None:
        if self._model is None:
            return
        import gc

        import torch

        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class VllmOcrEngine:
    def __init__(self, base_url: str, model: str, transport: httpx.BaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._transport = transport

    def _client(self, timeout: float) -> httpx.Client:
        return httpx.Client(timeout=timeout, transport=self._transport)

    def availability(self) -> tuple[bool, str]:
        try:
            with self._client(3) as client:
                resp = client.get(f"{self.base_url}/models")
            if resp.is_success:
                return True, ""
        except httpx.HTTPError:
            pass
        return False, VLLM_UNAVAILABLE

    def ocr(self, image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": data_url}}],
                }
            ],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_NEW_TOKENS,
        }
        try:
            with self._client(300) as client:
                resp = client.post(f"{self.base_url}/chat/completions", json=body)
                resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as e:
            raise ParseError(f"OCR 服務（vLLM）呼叫失敗：{e}") from e

    def release(self) -> None:
        pass


def build_ocr_engine(settings: Settings) -> OcrEngine:
    if settings.ocr_backend == "vllm":
        return VllmOcrEngine(settings.vllm_base_url, settings.ocr_model)
    return TransformersOcrEngine(settings.ocr_model, settings.ollama_base_url)

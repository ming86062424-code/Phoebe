import importlib.util
import logging

import httpx

from app.kb.parsers.base import PageText, ParseError, ProgressCallback

logger = logging.getLogger(__name__)

RENDER_DPI = 400
MAX_EDGE = 2048
MAX_NEW_TOKENS = 4096


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


class LightOnOcrParser:
    id = "lightonocr"
    label = "LightOnOCR-3（GPU OCR，適合掃描檔與複雜版面）"

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
            # Turing and older GPUs lack native bf16; fp16 is much faster there.
            major, _ = torch.cuda.get_device_capability()
            dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            device, dtype = "cpu", torch.float32
        self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_id, dtype=dtype
        ).to(device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def _ocr_image(self, image) -> str:
        import torch

        conversation = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        inputs = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.2
            )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        return self._processor.decode(new_tokens, skip_special_tokens=True).strip()

    def parse(self, pdf_path: str, on_progress: ProgressCallback) -> list[PageText]:
        import pypdfium2 as pdfium

        try:
            self._ensure_loaded()
            pdf = pdfium.PdfDocument(pdf_path)
        except Exception as e:
            raise ParseError(f"LightOnOCR 初始化失敗：{e}") from e
        try:
            total = len(pdf)
            on_progress(0, total)
            pages: list[PageText] = []
            for i in range(total):
                page = pdf[i]
                width, height = page.get_size()
                scale = min(RENDER_DPI / 72, MAX_EDGE / max(width, height))
                image = page.render(scale=scale).to_pil().convert("RGB")
                try:
                    text = self._ocr_image(image)
                except Exception as e:
                    raise ParseError(f"第 {i + 1} 頁 OCR 失敗：{e}") from e
                pages.append(PageText(i + 1, text))
                on_progress(i + 1, total)
            return pages
        finally:
            pdf.close()

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

import json

import httpx
import pytest
from PIL import Image

from app.config import Settings
from app.kb.parsers.base import ParseError
from app.kb.parsers.ocr_engines import (
    TransformersOcrEngine,
    VllmOcrEngine,
    build_ocr_engine,
)

BASE = "http://vllm:8001/v1"


def _engine(handler):
    return VllmOcrEngine(BASE, "ocr-model", transport=httpx.MockTransport(handler))


def test_vllm_ocr_sends_png_data_url_and_parses_reply():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "辨識結果"}}]})

    text = _engine(handler).ocr(Image.new("RGB", (8, 8), "white"))
    assert text == "辨識結果"
    assert seen["path"] == "/v1/chat/completions"
    body = seen["body"]
    assert body["model"] == "ocr-model"
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.9
    assert body["max_tokens"] == 4096
    part = body["messages"][0]["content"][0]
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")


def test_vllm_ocr_http_error_raises_parse_error():
    engine = _engine(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(ParseError, match="OCR 服務（vLLM）呼叫失敗"):
        engine.ocr(Image.new("RGB", (8, 8)))


def test_vllm_availability():
    ok = _engine(lambda request: httpx.Response(200, json={"data": []}))
    assert ok.availability() == (True, "")

    def refuse(request):
        raise httpx.ConnectError("refused")

    assert _engine(refuse).availability() == (False, "OCR 服務（vLLM）未啟動")
    bad = _engine(lambda request: httpx.Response(503))
    assert bad.availability() == (False, "OCR 服務（vLLM）未啟動")


def test_build_ocr_engine_selects_backend(monkeypatch):
    monkeypatch.setenv("OCR_BACKEND", "transformers")
    assert isinstance(build_ocr_engine(Settings.from_env()), TransformersOcrEngine)
    monkeypatch.setenv("OCR_BACKEND", "vllm")
    assert isinstance(build_ocr_engine(Settings.from_env()), VllmOcrEngine)

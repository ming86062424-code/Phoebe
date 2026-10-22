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
    fix_escaped_dollars,
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
    assert body["temperature"] == 0.1
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "top_p" not in body
    assert body["max_tokens"] == 4096
    # official grounding request: the page image, then the text "grounding"
    part, prompt = body["messages"][0]["content"]
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")
    assert prompt == {"type": "text", "text": "grounding"}


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


def test_fix_escaped_dollars_repairs_math_but_not_prices():
    assert fix_escaped_dollars(r"設 \$x^2\$ 為正") == "設 $x^2$ 為正"
    assert fix_escaped_dollars(r"\$\$a+b\$\$") == "$$a+b$$"
    block = "a\n" + r"\$\$" + "\nx = 1\n" + r"\$\$"
    assert fix_escaped_dollars(block) == "a\n$$\nx = 1\n$$"
    # prices and lone escapes stay as they are
    assert fix_escaped_dollars(r"NT\$ 1,280") == r"NT\$ 1,280"
    assert fix_escaped_dollars(r"\$5") == r"\$5"
    assert fix_escaped_dollars(r"\$5 and \$6") == r"\$5 and \$6"
    assert fix_escaped_dollars("no dollars") == "no dollars"
    # LaTeX commands inside inline math
    assert fix_escaped_dollars(r"\$\alpha+1\$") == r"$\alpha+1$"
    assert fix_escaped_dollars(r"\$\frac{a}{b}\$") == r"$\frac{a}{b}$"
    assert fix_escaped_dollars(r"\$\alpha\$ 與 \$\beta\$") == r"$\alpha$ 與 $\beta$"
    # price ranges are not pairs
    assert fix_escaped_dollars(r"\$5-\$10") == r"\$5-\$10"
    assert fix_escaped_dollars(r"NT\$1,280 to NT\$2,000") == r"NT\$1,280 to NT\$2,000"
    assert fix_escaped_dollars("NT$ 1,280") == "NT$ 1,280"


def test_vllm_ocr_output_is_repaired():
    engine = _engine(
        lambda request: httpx.Response(
            200, json={"choices": [{"message": {"content": r"\$x\$"}}]}
        )
    )
    assert engine.ocr(Image.new("RGB", (8, 8))) == "$x$"


def test_vllm_engine_uses_the_configured_timeout():
    seen = []

    def handler(request):
        seen.append((request.url.path, request.extensions["timeout"]["read"]))
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    transport = httpx.MockTransport(handler)
    VllmOcrEngine(BASE, "m", transport=transport).ocr(Image.new("RGB", (8, 8)))
    VllmOcrEngine(BASE, "m", transport=transport, timeout=42).ocr(Image.new("RGB", (8, 8)))
    VllmOcrEngine(BASE, "m", transport=transport, timeout=42).availability()
    assert seen == [
        ("/v1/chat/completions", 600),
        ("/v1/chat/completions", 42),
        ("/v1/models", 3),  # the availability check stays short
    ]


def test_build_ocr_engine_passes_the_ocr_timeout(monkeypatch):
    monkeypatch.setenv("OCR_BACKEND", "vllm")
    monkeypatch.setenv("OCR_TIMEOUT_SECONDS", "120")
    assert build_ocr_engine(Settings.from_env()).timeout == 120

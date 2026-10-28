"""RAG quality on the running system (docs/rag-eval.md).

Run only by hand, with the Docker services up:
    uv run --group eval pytest evals/test_rag_quality.py
"""

import time
from pathlib import Path

import httpx
import pytest
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase

from evals.metrics import rag_answer_metrics
from evals.rag_app import RAG_DIR, LiveRag, Reply
from evals.rag_report import short_error, source_rank, with_retry, write_report
from scripts.benchmark import _git_commit

HERE = Path(__file__).resolve().parent
BASE_URL = "http://localhost:8000"
REPORT_DIR = HERE / "reports"

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path=str(HERE / "rag.dataset.json"))
GOLDENS = dataset.goldens
RETRIEVAL_GOLDENS = [g for g in GOLDENS if g.additional_metadata["sources"]]

records: list[dict] = []
_replies: dict[str, Reply] = {}


def _ids(goldens: list[Golden]) -> list[str]:
    return [g.name for g in goldens]


@pytest.fixture(scope="session")
def rag(config_value, settings):
    user, password = config_value("ADMIN_USERNAME"), config_value("ADMIN_PASSWORD")
    if not user or not password:
        pytest.skip("沒有設定 ADMIN_USERNAME／ADMIN_PASSWORD（docker/.env）")
    if not RAG_DIR.exists():
        pytest.skip(f"找不到測試資料 {RAG_DIR}")
    try:
        live = LiveRag(BASE_URL, user, password)
    except httpx.HTTPError as e:
        pytest.skip(f"無法登入 {BASE_URL}（Docker 服務有開嗎？）：{e}")
    started = time.strftime("%Y-%m-%d %H:%M")
    try:
        statuses = live.setup()
        failed = [name for name, status in statuses.items() if status != "ready"]
        if failed:
            pytest.fail(f"文件處理失敗：{'、'.join(failed)}")
        environment = {
            "started": started,
            "git commit": _git_commit(),
            **live.environment(),
            "裁判模型": config_value("EVAL_JUDGE_MODEL", settings.eval_judge_model),
            "題庫": f"{len(GOLDENS)} 題（evals/rag.dataset.json），{len(statuses)} 份 md（md 一律用文字解析）",
        }
        yield live
        if records:
            path = write_report(REPORT_DIR, environment, records)
            print(f"\n報告：{path}")
    finally:
        live.cleanup()


def _reply(rag: LiveRag, golden: Golden) -> Reply:
    """The chat's reply, asked once per question (both tests use it). A failure becomes the
    reply's error, so both tests still record the question."""
    if golden.input not in _replies:
        try:
            _replies[golden.input] = with_retry(lambda: rag.ask(golden.input))
        except Exception as e:
            _replies[golden.input] = Reply(error=short_error(e))
    return _replies[golden.input]


def _source_name(s: dict) -> str:
    heading = next((line[3:] for line in s.get("text", "").split("\n") if line.startswith("## ")), "")
    return f"{Path(s['filename']).stem}／{heading}" if heading else s["filename"]


@pytest.mark.parametrize("golden", RETRIEVAL_GOLDENS, ids=_ids(RETRIEVAL_GOLDENS))
def test_retrieval(rag, golden: Golden):
    """The expected source is among the chunks the chat retrieved (no judge involved)."""
    meta = golden.additional_metadata
    reply = _reply(rag, golden)
    rank = source_rank(reply.sources, meta["sources"])
    records.append({
        "name": golden.name, "category": meta["category"], "kind": "retrieval",
        "question": golden.input, "rank": rank, "expected": meta["sources"],
        "retrieved": [_source_name(s) for s in reply.sources], "error": reply.error,
    })
    assert reply.error is None, reply.error
    assert rank is not None, f"應出現的來源不在前 {len(reply.sources)} 段"


@pytest.mark.parametrize("golden", GOLDENS, ids=_ids(GOLDENS))
def test_answer(rag, judge, golden: Golden):
    """The answer, scored by the judge model."""
    meta = golden.additional_metadata
    reply = _reply(rag, golden)
    record = {
        "name": golden.name, "category": meta["category"], "kind": "answer",
        "question": golden.input, "answer": reply.answer,
        "expected_output": golden.expected_output, "metrics": {}, "error": reply.error,
    }
    records.append(record)
    assert reply.error is None, reply.error
    metrics = rag_answer_metrics(judge, meta["category"], bool(golden.expected_output))
    case = LLMTestCase(
        input=golden.input,
        actual_output=reply.answer,
        expected_output=golden.expected_output,
        retrieval_context=[s["text"] for s in reply.sources] or ["（知識庫中沒有找到相關內容）"],
    )
    for m in metrics:  # one at a time: the judge's per-minute quota is small
        try:
            with_retry(lambda m=m: m.measure(case, _show_indicator=False))
            result = {"score": m.score, "success": bool(m.success), "reason": m.reason}
        except Exception as e:
            result = {"score": None, "success": False, "reason": f"評分失敗：{short_error(e)}"}
        record["metrics"][m.__name__] = result
    failed = {name: r["score"] for name, r in record["metrics"].items() if not r["success"]}
    assert not failed, f"沒通過：{failed}"

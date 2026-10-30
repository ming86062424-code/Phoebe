import json
from pathlib import Path

import pytest

from evals.rag_report import (
    CATEGORY_LABEL,
    RETRIEVAL_ONLY,
    render_markdown,
    source_rank,
    summarize,
    write_report,
)

LEAVE = {"file": "01_人事規章/請假制度.md", "section": "特別休假的計算與遞延"}


def chunk(filename: str, section: str, title: str = "") -> dict:
    head = f"# {title}\n\n" if title else ""
    return {"filename": filename, "text": f"{head}## {section}\n\n內容"}


def test_source_rank_matches_file_and_heading_line():
    sources = [
        chunk("薪資與獎金.md", "特別休假的計算與遞延"),  # right heading, wrong file
        chunk("請假制度.md", "病假與住院病假"),  # right file, wrong section
        chunk("請假制度.md", "特別休假的計算與遞延", title="星澄科技 請假制度"),
    ]
    assert source_rank(sources, [LEAVE]) == 3


def test_source_rank_needs_the_whole_heading():
    # 「## 特別休假」 is only a prefix of the expected heading
    assert source_rank([chunk("請假制度.md", "特別休假")], [LEAVE]) is None
    assert source_rank([], [LEAVE]) is None


def test_source_rank_any_of_several_expected_sources():
    other = {"file": "01_人事規章/績效考核.md", "section": "等第分布與校準"}
    assert source_rank([chunk("績效考核.md", "等第分布與校準")], [LEAVE, other]) == 1


def _records():
    return [
        {"name": "g1", "category": "general", "kind": "retrieval", "rank": 1},
        {"name": "g2", "category": "general", "kind": "retrieval", "rank": 2},
        {"name": "k1", "category": "keyword", "kind": "retrieval", "rank": None,
         "question": "FL|360C", "expected": [LEAVE], "retrieved": ["a／b"]},
        {"name": "g1", "category": "general", "kind": "answer", "question": "q", "answer": "a",
         "metrics": {"Faithfulness": {"score": 1.0, "success": True, "reason": "ok"}}},
        {"name": "n1", "category": "no_answer", "kind": "answer", "question": "q2",
         "answer": "知識庫\n沒有", "expected_output": None,
         "metrics": {"Faithfulness": {"score": 0.5, "success": False, "reason": "made up"},
                     "承認沒有答案 [GEval]": {"score": None, "success": False, "reason": "timeout"}}},
    ]


def test_summarize_per_category_and_all():
    s = summarize(_records())
    assert list(s) == ["general", "keyword", "no_answer", "all"]  # dataset order, then all
    assert s["general"]["retrieval"] == {"questions": 2, "hit_rate": 1.0, "top1_rate": 0.5, "mrr": 0.75}
    assert s["keyword"]["retrieval"]["hit_rate"] == 0.0
    assert s["all"]["retrieval"]["mrr"] == 0.5  # (1 + 0.5 + 0) / 3
    assert "retrieval" not in s["no_answer"]
    assert s["all"]["metrics"]["Faithfulness"] == {"questions": 2, "mean": 0.75, "pass_rate": 0.5}
    # a metric that errored has no score: left out of the mean, counted as not passed
    assert s["no_answer"]["metrics"]["承認沒有答案 [GEval]"] == {"questions": 1, "mean": None, "pass_rate": 0.0}


def test_render_lists_misses_and_failures():
    records = _records()
    md = render_markdown({"started": "2026-10-27 10:00", "裁判模型": "gemini"}, records, summarize(records))
    assert "- 裁判模型：gemini" in md
    assert "| 一般問答 | 2 | 100% | 50% | 0.75 |" in md
    assert "FL\\|360C" in md  # a | in a question cannot break the table
    assert "請假制度／特別休假的計算與遞延" in md
    assert "### n1：q2" in md and "**回答**：知識庫 沒有" in md
    assert "- Faithfulness（0.50）：made up" in md
    assert "### g1" not in md  # passed every metric


def test_write_report_md_and_json(tmp_path):
    md = write_report(tmp_path / "reports", {"started": "x"}, _records())
    assert md.suffix == ".md" and md.name.startswith("rag-")
    data = json.loads(md.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["summary"]["general"]["retrieval"]["questions"] == 2
    assert len(data["records"]) == 5


GEMINI_429 = ("429 RESOURCE_EXHAUSTED. {'error': {'message': 'Quota exceeded ... "
              "Please retry in 21.83s.'}}")


def test_retry_delay_reads_gemini_hint_and_phoebe_503():
    from evals.rag_report import RATE_LIMIT_WAIT, retry_delay

    assert retry_delay(GEMINI_429) == 22.83
    assert retry_delay("對話失敗 503：{\"detail\":\"Gemini 配額已用完或請求太頻繁，請稍後再試\"}") == RATE_LIMIT_WAIT
    assert retry_delay("對話失敗 503：知識庫檢索失敗") is None


def test_with_retry_waits_on_rate_limit_only():
    from evals.rag_report import with_retry

    waits, calls = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError(GEMINI_429)
        return "ok"

    assert with_retry(flaky, sleep=waits.append) == "ok"
    assert waits == [22.83, 22.83]

    def broken():
        raise ValueError("not a rate limit")

    with pytest.raises(ValueError):
        with_retry(broken, sleep=waits.append)
    assert len(waits) == 2  # no wait for other errors

    def always_limited():
        raise RuntimeError(GEMINI_429)

    with pytest.raises(RuntimeError):
        with_retry(always_limited, tries=3, sleep=waits.append)
    assert len(waits) == 4  # waited between the 3 tries, then gave up


GEMINI_DAILY = ("429 RESOURCE_EXHAUSTED. {'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', "
                "'message': 'Please retry in 17h26m58.96s.'}")


def test_daily_limit_is_not_retried():
    from evals.rag_report import retry_delay, short_error, with_retry

    assert retry_delay(GEMINI_DAILY) is None
    waits = []

    def daily():
        raise RuntimeError(GEMINI_DAILY)

    with pytest.raises(RuntimeError):
        with_retry(daily, sleep=waits.append)
    assert waits == []
    assert short_error(RuntimeError(GEMINI_DAILY)) == "Gemini 免費方案今日的請求次數已用完"
    assert short_error(RuntimeError("x" * 300)).endswith("…")


def test_dataset_categories_and_answers():
    """Every question has a known category; the bare codes and short phrases only test
    retrieval, so they carry sources but no expected answer."""
    path = Path(__file__).resolve().parent.parent / "evals" / "rag.dataset.json"
    goldens = json.loads(path.read_text(encoding="utf-8"))
    assert len({g["name"] for g in goldens}) == len(goldens)
    for g in goldens:
        meta = g["additional_metadata"]
        assert meta["category"] in CATEGORY_LABEL, g["name"]
        if meta["category"] in RETRIEVAL_ONLY:
            assert meta["sources"] and not g.get("expected_output"), g["name"]
        elif meta["category"] == "no_answer":
            assert not meta["sources"], g["name"]
        else:
            assert meta["sources"] and g.get("expected_output"), g["name"]

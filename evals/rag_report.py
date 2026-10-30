"""Scoring, rate-limit retries and the report of the RAG eval (evals/test_rag_quality.py).

No deepeval import, so the unit tests in tests/ can cover all of it.
"""

import json
import re
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")

# Gemini's free tier allows 15 requests a minute per model; its 429 says how long to wait
RATE_LIMIT_WAIT = 65.0
RATE_LIMIT_TRIES = 6
_RATE_LIMITED = ("RESOURCE_EXHAUSTED", "配額已用完或請求太頻繁")  # Gemini's 429, Phoebe's 503
# the free tier's daily cap (500 a day per model) resets the next day: waiting cannot help
_DAILY_LIMIT = ("PerDay", "per day")

CATEGORY_LABEL = {
    "general": "一般問答",
    "keyword": "關鍵字（只輸入代號）",
    "keyword_question": "關鍵字（完整問題）",
    "semantic_phrase": "語意（簡短詞組）",
    "semantic": "語意（完整問題）",
    "confusable": "容易混淆",
    "no_answer": "知識庫沒有答案",
}
# a bare code or a short phrase only checks what retrieval finds: there is no question to answer
RETRIEVAL_ONLY = frozenset({"keyword", "semantic_phrase"})


def retry_delay(message: str) -> float | None:
    """Seconds to wait before retrying a rate-limited call, None when it was not rate limited."""
    if not any(mark in message for mark in _RATE_LIMITED):
        return None
    if any(mark in message for mark in _DAILY_LIMIT):
        return None
    hint = re.search(r"retry in ([\d.]+)s", message)
    return float(hint.group(1)) + 1 if hint else RATE_LIMIT_WAIT


def with_retry(call: Callable[[], T], tries: int = RATE_LIMIT_TRIES,
               sleep: Callable[[float], None] = time.sleep) -> T:
    """call(), waiting and trying again while it fails for a rate limit; other errors raise."""
    for attempt in range(1, tries + 1):
        try:
            return call()
        except Exception as e:
            delay = retry_delay(str(e))
            if delay is None or attempt == tries:
                raise
            sleep(delay)
    raise AssertionError("unreachable")


def source_rank(sources: list[dict], expected: list[dict]) -> int | None:
    """1-based rank of the first retrieved chunk that is one of the expected sources, or None.

    A chunk matches when its file name is the expected file's name and its text has the
    expected `## section` heading line (a markdown chunk starts with its section heading).
    """
    for rank, source in enumerate(sources, start=1):
        text = source.get("text", "")
        for want in expected:
            name = Path(want["file"]).name
            if source.get("filename") == name and f"## {want['section']}" in text.split("\n"):
                return rank
    return None


def summarize(records: list[dict]) -> dict:
    """Per category: retrieval hit rate and MRR, and per metric its mean score and pass rate.

    A record is {"name", "category", "kind": "retrieval"|"answer", ...}:
    retrieval records carry "rank" (int or None); answer records carry
    "metrics": {metric name: {"score", "success", "reason"}}.
    """
    out: dict[str, dict] = {}
    categories = [c for c in CATEGORY_LABEL if any(r["category"] == c for r in records)]
    for category in [*categories, "all"]:
        rows = [r for r in records if category == "all" or r["category"] == category]
        retrieval = [r for r in rows if r["kind"] == "retrieval"]
        summary: dict = {}
        if retrieval:
            ranks = [r["rank"] for r in retrieval]
            summary["retrieval"] = {
                "questions": len(ranks),
                "hit_rate": sum(r is not None for r in ranks) / len(ranks),
                "top1_rate": sum(r == 1 for r in ranks) / len(ranks),
                "mrr": sum(1 / r for r in ranks if r) / len(ranks),
            }
        metrics: dict[str, list[dict]] = {}
        for r in rows:
            if r["kind"] == "answer":
                for name, m in r["metrics"].items():
                    metrics.setdefault(name, []).append(m)
        summary["metrics"] = {
            name: {
                "questions": len(ms),
                "mean": _mean([m["score"] for m in ms]),
                "pass_rate": sum(bool(m["success"]) for m in ms) / len(ms),
            }
            for name, ms in metrics.items()
        }
        out[category] = summary
    return out


def _mean(values: list[float | None]) -> float | None:
    scored = [v for v in values if v is not None]
    return sum(scored) / len(scored) if scored else None


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.0f}%"


def _score(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}"


def _label(category: str) -> str:
    return "全部" if category == "all" else CATEGORY_LABEL.get(category, category)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def short_error(e: Exception, limit: int = 200) -> str:
    """An error for the report: Gemini's 429 runs to a page of JSON."""
    text = " ".join(str(e).split())
    if any(mark in text for mark in _DAILY_LIMIT):
        return "Gemini 免費方案今日的請求次數已用完"
    return text if len(text) <= limit else text[:limit] + "…"


def render_markdown(environment: dict, records: list[dict], summary: dict) -> str:
    lines = [f"# RAG 評測 {environment.get('started', '')}", "", "## 環境", ""]
    lines += [f"- {k}：{v}" for k, v in environment.items() if k != "started"]

    lines += ["", "## 檢索（不需裁判模型）", "",
              "| 類別 | 題數 | 前 5 名命中 | 第 1 名命中 | MRR |", "|---|---|---|---|---|"]
    for category, s in summary.items():
        if r := s.get("retrieval"):
            lines.append(f"| {_label(category)} | {r['questions']} | {_pct(r['hit_rate'])} | "
                         f"{_pct(r['top1_rate'])} | {r['mrr']:.2f} |")

    names = sorted({n for s in summary.values() for n in s["metrics"]})
    if names:
        lines += ["", "## 回答（裁判模型評分，平均分數／通過率）", "",
                  "| 類別 | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for category, s in summary.items():
            cells = []
            for n in names:
                m = s["metrics"].get(n)
                cells.append("—" if m is None else f"{_score(m['mean'])}／{_pct(m['pass_rate'])}")
            lines.append(f"| {_label(category)} | " + " | ".join(cells) + " |")

    missed = [r for r in records if r["kind"] == "retrieval" and r["rank"] is None]
    if missed:
        lines += ["", "## 檢索沒命中的題目", "", "| 題目 | 問題 | 應出現的來源 | 實際前 5 名 |", "|---|---|---|---|"]
        for r in missed:
            want = "、".join(f"{Path(s['file']).stem}／{s['section']}" for s in r["expected"])
            got = "、".join(r["retrieved"]) or "（沒有結果）"
            lines.append(f"| {r['name']} | {_cell(r['question'])} | {_cell(want)} | {_cell(got)} |")

    failed = [r for r in records if r["kind"] == "answer"
              and any(not m["success"] for m in r["metrics"].values())]
    if failed:
        lines += ["", "## 回答沒通過的題目", ""]
        for r in failed:
            lines += [f"### {r['name']}：{r['question']}", "", f"**回答**：{_cell(r['answer'])}", ""]
            if r.get("expected_output"):
                lines += [f"**標準答案**：{r['expected_output']}", ""]
            for name, m in r["metrics"].items():
                if not m["success"]:
                    lines.append(f"- {name}（{_score(m['score'])}）：{_cell(m.get('reason') or '')}")
            lines.append("")
    if errors := [r for r in records if r.get("error")]:
        lines += ["", "## 執行錯誤", ""]
        lines += [f"- {r['name']}：{_cell(r['error'])}" for r in errors]
    return "\n".join(lines).rstrip() + "\n"


def write_report(out_dir: Path, environment: dict, records: list[dict]) -> Path:
    """Write <timestamp>.md and .json into out_dir; returns the markdown path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    summary = summarize(records)
    md = out_dir / f"rag-{stamp}.md"
    md.write_text(render_markdown(environment, records, summary), encoding="utf-8", newline="\n")
    data = {"environment": environment, "summary": summary, "records": records}
    md.with_suffix(".json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return md

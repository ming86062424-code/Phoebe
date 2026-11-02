import argparse

import pytest

from scripts.loadtest import MAX_USERS, Result, parse_steps, percentile, render_markdown, summarize_step


def test_parse_steps():
    assert parse_steps("1,2,5,10") == [1, 2, 5, 10]
    assert parse_steps(" 3 ") == [3]


@pytest.mark.parametrize("bad", ["", "0,2", f"1,{MAX_USERS + 1}", "5,2", "2,2", "a"])
def test_parse_steps_rejects(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_steps(bad)


def test_percentile_nearest_rank():
    values = [float(v) for v in range(1, 21)]  # 1..20
    assert percentile(values, 95) == 19
    assert percentile(values, 50) == 10
    assert percentile([4.0], 95) == 4
    assert percentile([], 95) is None


def test_summarize_step_counts_errors_and_times_only_successes():
    results = [
        Result(0, True, 1.0, 5.0),
        Result(1, True, 2.0, 7.0),
        Result(2, False, None, 0.5, "503 Gemini 配額已用完或請求太頻繁"),
        Result(3, False, None, 0.4, "503 Gemini 配額已用完或請求太頻繁"),
    ]
    s = summarize_step(4, results, wall=30.0)
    assert s["requests"] == 4 and s["ok"] == 2
    assert s["error_rate"] == 0.5
    assert s["total_p50"] == 6.0 and s["total_p95"] == 7.0
    assert s["first_byte_p50"] == 1.5
    assert s["per_minute"] == 4.0
    assert s["errors"] == {"503 Gemini 配額已用完或請求太頻繁": 2}


def test_summarize_step_all_failed():
    s = summarize_step(1, [Result(0, False, None, 1.0, "x")], wall=1.0)
    assert s["total_p95"] is None and s["error_rate"] == 1.0


def test_render_markdown():
    step = summarize_step(2, [Result(0, True, 1.0, 5.0), Result(1, False, None, 1.0, "boom")], 10.0)
    step["peaks"] = {"phoebe-web-1": 300, "全部容器": 6000}
    md = render_markdown({"started": "2026-11-03 10:00", "聊天模型": "gemini x"}, [step])
    assert "| 2 | 2 | 50% |" in md
    assert "5.0 秒" in md
    assert "boom（1 次）" in md
    assert "記憶體高峰" in md

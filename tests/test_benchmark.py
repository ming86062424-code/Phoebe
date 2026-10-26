"""The benchmark script's arithmetic (the script itself only runs on request)."""

import pytest

from scripts.benchmark import TOTAL, compare, peaks_in_window, render_markdown, summarize


def _run(duration, phases, peaks=None):
    return {"duration": duration, "phases": phases, "peaks": peaks or {}}


def test_peaks_in_window_per_source_and_total():
    samples = [
        {"ts": 10, "source": "phoebe-web-1", "kind": "container", "mem_mib": 200},
        {"ts": 10, "source": "phoebe-embedder-1", "kind": "container", "mem_mib": 2000},
        {"ts": 15, "source": "phoebe-web-1", "kind": "container", "mem_mib": 300},
        {"ts": 15, "source": "phoebe-embedder-1", "kind": "container", "mem_mib": 1800},
        {"ts": 15, "source": "GPU", "kind": "gpu", "mem_mib": 4000},
        {"ts": 99, "source": "phoebe-web-1", "kind": "container", "mem_mib": 9000},  # outside
    ]
    peaks = peaks_in_window(samples, 10, 20)
    assert peaks["phoebe-web-1"] == 300
    assert peaks["phoebe-embedder-1"] == 2000
    assert peaks["GPU"] == 4000
    assert peaks[TOTAL] == 2200  # the GPU is not a container
    assert peaks_in_window(samples, 50, 60) == {}


def test_summarize_separates_the_cold_run():
    s = summarize([
        _run(30, {"parse": 20, "embed": 10}, {"GPU": 3000}),
        _run(10, {"parse": 6, "embed": 4}, {"GPU": 2000}),
        _run(12, {"parse": 8, "embed": 4}, {"GPU": 2500}),
    ])
    assert s["runs"] == 3
    assert s["cold"] == {"duration": 30, "phases": {"embed": 10, "parse": 20}, "peaks": {"GPU": 3000}}
    assert s["warm"]["duration"] == 11
    assert s["warm"]["phases"] == {"embed": 4, "parse": 7}
    assert list(s["warm"]["phases"]) == ["parse", "embed"]  # in the order they happen
    assert s["warm"]["peaks"] == {"GPU": 2500}  # the highest of the warm runs
    one = summarize([_run(5, {"parse": 5})])
    assert one["cold"] == one["warm"]


def _report(duration, parse, peak, device="cpu"):
    return {
        "environment": {"commit": "abc", "started_at": "now", "embedder_device": device,
                        "other_containers_mib": 0},
        "threshold": 0.25,
        "tasks": {"文件處理：a.pdf（liteparse）": summarize([
            _run(duration, {"parse": parse, "store": 0.1}, {"phoebe-embedder-1": peak}),
        ])},
    }


def test_compare_flags_only_real_regressions():
    base = _report(100, 60, 2700)
    same = compare(_report(110, 70, 2800), base)  # +10 %: within the threshold
    assert same == {"slower": [], "higher": [], "environment": []}
    worse = compare(_report(140, 60, 3600), base)
    assert [(w["metric"], w["change"]) for w in worse["slower"]] == [("總時間", 0.4)]
    assert [w["metric"] for w in worse["higher"]] == ["embedder 記憶體高峰"]
    # a tiny phase may double without being flagged (0.1 s -> 0.3 s)
    tiny = _report(100, 60, 2700)
    tiny["tasks"]["文件處理：a.pdf（liteparse）"]["warm"]["phases"]["store"] = 0.3
    assert compare(tiny, base)["slower"] == []


def test_compare_reports_a_different_environment():
    diff = compare(_report(100, 60, 2700, device="cuda"), _report(100, 60, 2700))
    assert diff["environment"] == [{"item": "embedder_device", "before": "cpu", "now": "cuda"}]


@pytest.mark.parametrize("with_baseline", [True, False])
def test_render_markdown(with_baseline):
    report = _report(140, 60, 3600)
    comparison = compare(report, _report(100, 60, 2700)) if with_baseline else None
    text = render_markdown(report, comparison, "baseline.json" if with_baseline else None)
    assert "### 文件處理：a.pdf（liteparse）（1 次）" in text
    assert "| 解析 | 1 分 0 秒 | 1 分 0 秒 |" in text
    assert "| embedder 記憶體高峰 | 3.52 GB | 3.52 GB |" in text
    if with_baseline:
        assert "⚠ 文件處理：a.pdf（liteparse）／總時間：1 分 40 秒 → 2 分 20 秒（+40%）" in text
    else:
        assert "沒有基準可以比較" in text

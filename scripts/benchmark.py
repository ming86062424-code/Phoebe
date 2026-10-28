"""Performance benchmark: run a fixed scenario through the running Docker stack and report
each task's phase timings and the memory/VRAM peaks, compared with a baseline.

Run it only when you want numbers (it takes minutes and loads the machine); it is not part
of the test suite. See docs/benchmarks.md.

    uv run python -m scripts.benchmark                    # every task, 3 runs each
    uv run python -m scripts.benchmark --runs 2 --skip-chat --parsers liteparse
    uv run python -m scripts.benchmark --set-baseline     # this run becomes the baseline

Timings come from the monitoring events (monitor.db) the system records anyway, and the
peaks from the monitor service's samples over each task's time window, so nothing is timed
twice. It logs in as the admin account from docker/.env and deletes everything it creates.
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
DOCKER_ENV = ROOT / "docker" / ".env"
DEFAULT_SCENARIO = ROOT / "scripts" / "benchmark_scenario.json"
DEFAULT_OUT = ROOT / "docs" / "benchmarks"
# settings that change the numbers; recorded with every report (never secrets)
SETTING_KEYS = (
    "INGEST_CONCURRENCY",
    "OCR_CONCURRENCY",
    "OCR_VLLM_MAX_NUM_SEQS",
    "OCR_VLLM_KV_CACHE_MEMORY",
    "EMBED_IMAGE_TOKENS",
    "KB_TOP_K",
    "KB_MAX_IMAGES",
    "CHAT_THINK",
)
DOC_TIMEOUT = 1800
EVENT_WAIT = 15
DEFAULT_THRESHOLD = 0.25
TOTAL = "全部容器"
# the markdown report shows these peaks (the JSON keeps every source)
REPORT_SOURCES = ("phoebe-embedder-1", "phoebe-ingest-1", "phoebe-web-1", "phoebe-milvus-1",
                  "phoebe-vllm-1", "GPU", TOTAL)
PHASE_LABEL = {
    "parse": "解析", "embed": "向量", "store": "寫入", "search": "檢索", "images": "載入附圖",
    "first_token": "第一個回應", "thinking": "思考", "answer": "回答",
}


# ---- pure helpers (tests/test_benchmark.py) ----


def peaks_in_window(samples: list[dict], start: float, end: float) -> dict[str, float]:
    """The highest memory of every source within [start, end], plus the sum of all
    containers per sampling round ("全部容器")."""
    peaks: dict[str, float] = {}
    rounds: dict[float, float] = {}
    for s in samples:
        if not start <= s["ts"] <= end:
            continue
        peaks[s["source"]] = max(peaks.get(s["source"], 0.0), s["mem_mib"])
        if s["kind"] == "container":
            rounds[s["ts"]] = rounds.get(s["ts"], 0.0) + s["mem_mib"]
    if rounds:
        peaks[TOTAL] = max(rounds.values())
    return peaks


def summarize(runs: list[dict]) -> dict:
    """Runs of one task -> cold (the first run) and warm (the mean of the others; the
    first run when there is only one). Each run: {"duration", "phases", "peaks"}."""
    def collapse(group: list[dict]) -> dict:
        # in the order the phases happen (PHASE_LABEL's order), unknown ones last
        order = list(PHASE_LABEL)
        phases = sorted({p for r in group for p in r["phases"]},
                        key=lambda p: (order.index(p) if p in order else len(order), p))
        sources = sorted({s for r in group for s in r.get("peaks", {})})
        return {
            "duration": round(mean(r["duration"] for r in group), 2),
            "phases": {p: round(mean(r["phases"].get(p, 0.0) for r in group), 2) for p in phases},
            "peaks": {s: round(max(r["peaks"].get(s, 0.0) for r in group)) for s in sources},
        }

    warm = runs[1:] or runs[:1]
    return {"runs": len(runs), "cold": collapse(runs[:1]), "warm": collapse(warm)}


def compare(current: dict, baseline: dict, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Warm timings and peaks more than `threshold` worse than the baseline's, and the
    environment differences that make the comparison less meaningful."""
    slower, higher = [], []
    for task, now in current["tasks"].items():
        before = baseline.get("tasks", {}).get(task)
        if not before:
            continue
        pairs = [("總時間", now["warm"]["duration"], before["warm"]["duration"])] + [
            (PHASE_LABEL.get(p, p), v, before["warm"]["phases"][p])
            for p, v in now["warm"]["phases"].items()
            if p in before["warm"]["phases"]
        ]
        for label, v, old in pairs:
            # sub-second phases jitter by more than 25 %: only flag a real difference
            if old > 0 and v > old * (1 + threshold) and v - old >= 0.5:
                slower.append({"task": task, "metric": label, "before": old, "now": v,
                               "change": round(v / old - 1, 3)})
        for source, v in now["warm"]["peaks"].items():
            old = before["warm"]["peaks"].get(source)
            if old and v > old * (1 + threshold) and v - old >= 256:
                higher.append({"task": task, "metric": f"{_source(source)} 記憶體高峰", "before": old,
                               "now": v, "change": round(v / old - 1, 3)})
    differences = [
        {"item": key, "before": baseline.get("environment", {}).get(key), "now": value}
        for key, value in current["environment"].items()
        if key not in ("commit", "started_at", "other_containers_mib")
        and baseline.get("environment", {}).get(key) != value
    ]
    return {"slower": slower, "higher": higher, "environment": differences}


def _seconds(v: float) -> str:
    if v < 1:
        return f"{v * 1000:.0f} ms"
    return f"{v:.1f} 秒" if v < 60 else f"{int(v // 60)} 分 {v % 60:.0f} 秒"


def _memory(mib: float) -> str:
    return f"{mib / 1024:.2f} GB" if mib >= 1024 else f"{mib:.0f} MB"


def _source(source: str) -> str:
    """phoebe-embedder-1 -> embedder"""
    return source.removeprefix("phoebe-").removesuffix("-1")


def render_markdown(report: dict, comparison: dict | None, baseline_name: str | None) -> str:
    env = report["environment"]
    lines = [f"# 效能測試 {env['started_at']}", "", "## 環境", "", "| 項目 | 值 |", "|---|---|"]
    for key, value in env.items():
        lines.append(f"| {key} | {value} |")
    lines += ["", "## 結果", "",
              "「冷啟動」是第一次執行，含模型載入與快取建立；「平均」是之後幾次的平均，比較時看這一欄。", ""]
    for task, s in report["tasks"].items():
        lines += [f"### {task}（{s['runs']} 次）", "", "| | 冷啟動 | 平均 |", "|---|---|---|",
                  f"| 總時間 | {_seconds(s['cold']['duration'])} | {_seconds(s['warm']['duration'])} |"]
        for p, v in s["warm"]["phases"].items():
            lines.append(f"| {PHASE_LABEL.get(p, p)} | {_seconds(s['cold']['phases'].get(p, 0))} | {_seconds(v)} |")
        for source in REPORT_SOURCES:
            if source in s["warm"]["peaks"]:
                lines.append(f"| {_source(source)} 記憶體高峰 | {_memory(s['cold']['peaks'].get(source, 0))}"
                             f" | {_memory(s['warm']['peaks'][source])} |")
        lines.append("")
    lines += ["## 與基準比較", ""]
    if comparison is None:
        lines.append("沒有基準可以比較（用 `--set-baseline` 把這次設為基準）。")
    else:
        lines.append(f"基準：`{baseline_name}`，超過基準 {int(report['threshold'] * 100)}% 列為警告。")
        lines.append("")
        if comparison["environment"]:
            lines += ["**環境和基準不同，比較僅供參考：**", ""]
            lines += [f"- {d['item']}：{d['before']} → {d['now']}" for d in comparison["environment"]]
            lines.append("")
        warnings = comparison["slower"] + comparison["higher"]
        if not warnings:
            lines.append("沒有超過門檻的項目。")
        for w in warnings:
            fmt = _memory if "記憶體" in w["metric"] else _seconds
            lines.append(f"- ⚠ {w['task']}／{w['metric']}：{fmt(w['before'])} → {fmt(w['now'])}（+{w['change'] * 100:.0f}%）")
    return "\n".join(lines) + "\n"


# ---- running the scenario against the stack ----


class Bench:
    def __init__(self, base_url: str, user: str, password: str):
        self.c = httpx.Client(base_url=base_url, timeout=600,
                              limits=httpx.Limits(max_keepalive_connections=0),
                              transport=httpx.HTTPTransport(retries=3))
        self.c.post("/api/auth/login", json={"username": user, "password": password}).raise_for_status()
        self.kbs: list[int] = []
        self.conversations: list[int] = []

    def get(self, path: str, **params):
        r = self.c.get(path, params=params)
        r.raise_for_status()
        return r.json()

    def create_kb(self, name: str, parser: str) -> int:
        kb = self.c.post("/api/kbs", json={"name": name, "parser": parser})
        kb.raise_for_status()
        self.kbs.append(kb.json()["id"])
        return kb.json()["id"]

    def delete_kb(self, kb_id: int) -> None:
        self.c.delete(f"/api/kbs/{kb_id}")
        if kb_id in self.kbs:
            self.kbs.remove(kb_id)

    def upload(self, kb_id: int, path: Path) -> dict:
        r = self.c.post(f"/api/kbs/{kb_id}/documents", files={"file": (path.name, path.read_bytes())})
        r.raise_for_status()
        doc_id = r.json()["id"]
        deadline = time.time() + DOC_TIMEOUT
        while time.time() < deadline:
            doc = next(d for d in self.get(f"/api/kbs/{kb_id}/documents") if d["id"] == doc_id)
            if doc["status"] in ("ready", "failed"):
                return doc
            time.sleep(1)
        raise TimeoutError(f"{path.name} 處理超過 {DOC_TIMEOUT} 秒")

    def event(self, kind: str, match: dict, since: float) -> dict:
        """The monitoring event of a task just finished (written right after it ends)."""
        deadline = time.time() + EVENT_WAIT
        while time.time() < deadline:
            for e in self.get("/api/monitor/events", kind=kind, limit=50)["events"]:
                if e["started_at"] >= since - 1 and all(e["details"].get(k) == v for k, v in match.items()):
                    return e
            time.sleep(1)
        raise RuntimeError(f"找不到 {kind} 的監控事件：{match}")

    def samples(self, start: float, end: float, interval: float) -> list[dict]:
        minutes = min(math.ceil((time.time() - start) / 60) + 1, 2880)
        sources = [s["source"] for s in self.get("/api/monitor/status")["latest"]]
        rows = []
        for source in sources:
            rows += self.get("/api/monitor/samples", minutes=minutes, source=source, limit=2000)["samples"]
        return [r for r in rows if start - interval <= r["ts"] <= end + interval]

    def chat(self, kb_id: int, question: str) -> int:
        cid = self.c.post("/api/conversations").json()["id"]
        self.conversations.append(cid)
        self.c.patch(f"/api/conversations/{cid}", json={"kb_id": kb_id}).raise_for_status()
        with self.c.stream("POST", f"/api/conversations/{cid}/messages", json={"content": question}) as r:
            r.raise_for_status()
            last = [line for line in r.iter_lines() if line][-1]
        if json.loads(last)["type"] != "done":
            raise RuntimeError(f"對話失敗：{last}")
        return cid

    def cleanup(self) -> None:
        for cid in list(self.conversations):
            self.c.delete(f"/api/conversations/{cid}")
        for kb_id in list(self.kbs):
            self.delete_kb(kb_id)


def _git_commit() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("（有未 commit 的修改）" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "無"


def environment(bench: Bench, env: dict, embedder_url: str) -> tuple[dict, dict]:
    status = bench.get("/api/monitor/status")
    if not status["running"]:
        print("⚠ monitor 服務沒有在取樣：報告不會有記憶體與顯存高峰", file=sys.stderr)
    containers = [s for s in status["latest"] if s["kind"] == "container"]
    others = sum(s["mem_mib"] for s in containers if s["project"] != "phoebe")
    try:
        device = httpx.get(f"{embedder_url}/health", timeout=10).json().get("device", "cpu")
    except (httpx.HTTPError, ValueError):
        device = "unknown"
    parsers = {p["id"]: p["available"] for p in bench.get("/api/parsers")}
    llm = bench.get("/api/settings")
    model = llm["gemini_model"] if llm["llm_provider"] == "gemini" else llm["ollama_model"]
    info = {
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "commit": _git_commit(),
        "mode": "GPU embedding 模式" if device == "cuda" else
                ("LightOnOCR 模式" if parsers.get("lightonocr") else "CPU（未啟動 vLLM）"),
        "embedder_device": device,
        "lightonocr_available": parsers.get("lightonocr", False),
        "llm": f"{llm['llm_provider']} {model}",
        "gpu": _gpu_name(),
        "docker_vm_memory": f"{max((s['mem_limit_mib'] or 0) for s in containers) / 1024:.1f} GB" if containers else "unknown",
        "other_containers_mib": round(others),
        **{k: env.get(k, "（預設）") for k in SETTING_KEYS},
    }
    return info, status


def run(args) -> int:
    scenario = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    env = dotenv_values(DOCKER_ENV)
    if not env.get("ADMIN_USERNAME") or not env.get("ADMIN_PASSWORD"):
        sys.exit("docker/.env 沒有設定 ADMIN_USERNAME／ADMIN_PASSWORD（腳本以管理員登入讀取監控資料）")
    bench = Bench(args.url, env["ADMIN_USERNAME"], env["ADMIN_PASSWORD"])
    info, status = environment(bench, env, args.embedder_url)
    interval = (status.get("collector") or {}).get("interval_seconds") or 5
    if info["other_containers_mib"] > 1024:
        print(f"⚠ 其他專案的容器佔了 {_memory(info['other_containers_mib'])}，數字會受影響（例如 RAGFlow）",
              file=sys.stderr)
    # txt/md always go through the text parser, whatever the knowledge base's setting
    available = {p["id"] for p in bench.get("/api/parsers") if p["available"]} | {"text"}
    tasks: dict[str, list[dict]] = {}
    stamp = datetime.now().strftime("%m%d-%H%M")

    def record(task: str, event: dict) -> None:
        # the monitor samples every `interval` seconds: wait for the first sample after the
        # task, or a short task (a search takes ~0.2 s) would have no sample at all
        time.sleep(max(0.0, event["ended_at"] + interval + 1 - time.time()))
        peaks = peaks_in_window(bench.samples(event["started_at"], event["ended_at"], interval),
                                event["started_at"], event["ended_at"] + interval)
        tasks.setdefault(task, []).append({"duration": event["duration"], "phases": event["phases"], "peaks": peaks})
        print(f"  {task}：{_seconds(event['duration'])}", flush=True)

    try:
        for item in scenario["documents"]:
            path = ROOT / item["file"]
            for parser in item["parsers"]:
                if args.parsers and parser not in args.parsers:
                    continue
                if parser not in available:
                    print(f"略過 {path.name}（{parser}）：解析器目前不可用", file=sys.stderr)
                    continue
                task = f"文件處理：{path.name}（{parser}）"
                for n in range(args.runs):
                    kb_id = bench.create_kb(f"benchmark-{stamp}", "liteparse" if parser == "text" else parser)
                    start = time.time()
                    doc = bench.upload(kb_id, path)
                    if doc["status"] != "ready":
                        raise RuntimeError(f"{path.name} 處理失敗：{doc['error']}")
                    record(task, bench.event("document", {"document_id": doc["id"]}, start))
                    bench.delete_kb(kb_id)

        kb_file = ROOT / scenario["knowledge_base_document"]
        kb_id = bench.create_kb(f"benchmark-{stamp}-qa", "liteparse")
        if bench.upload(kb_id, kb_file)["status"] != "ready":
            raise RuntimeError("問答用的知識庫建立失敗")
        for n in range(args.runs):
            for i, question in enumerate(scenario["questions"], start=1):
                start = time.time()
                bench.get(f"/api/kbs/{kb_id}/search", q=question)
                record(f"搜尋測試：問題 {i}", bench.event("search", {"kb_id": kb_id}, start))
                if not args.skip_chat:
                    start = time.time()
                    cid = bench.chat(kb_id, question)
                    record(f"AI 對話：問題 {i}", bench.event("chat", {"conversation_id": cid}, start))
    finally:
        bench.cleanup()

    report = {"environment": info, "threshold": args.threshold,
              "tasks": {task: summarize(runs) for task, runs in tasks.items()}}
    baseline_path = Path(args.compare) if args.compare else Path(args.out) / "baseline.json"
    comparison = None
    if baseline_path.is_file():
        comparison = compare(report, json.loads(baseline_path.read_text(encoding="utf-8")), args.threshold)
        report["comparison"] = comparison
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("%Y%m%d-%H%M")
    (out / f"{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"{name}.md").write_text(
        render_markdown(report, comparison, baseline_path.name if comparison else None), encoding="utf-8")
    print(f"\n報告：{out / (name + '.md')}")
    if args.set_baseline:
        shutil.copyfile(out / f"{name}.json", out / "baseline.json")
        print(f"已設為基準：{out / 'baseline.json'}")
    warnings = (comparison or {}).get("slower", []) + (comparison or {}).get("higher", [])
    for w in warnings:
        print(f"⚠ {w['task']}／{w['metric']}：+{w['change'] * 100:.0f}%")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="效能測試（只在需要時手動執行）")
    ap.add_argument("--runs", type=int, default=3, help="每個任務跑幾次（第一次算冷啟動，預設 3）")
    ap.add_argument("--parsers", nargs="+", help="只測這些解析器，例如 liteparse text")
    ap.add_argument("--skip-chat", action="store_true", help="不測 AI 對話（會呼叫聊天模型，雲端模型會計費）")
    ap.add_argument("--scenario", default=str(DEFAULT_SCENARIO), help="情境檔（預設 scripts/benchmark_scenario.json）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="報告目錄（預設 docs/benchmarks）")
    ap.add_argument("--compare", help="比較的基準 JSON（預設報告目錄裡的 baseline.json）")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="比基準差多少算警告（預設 0.25）")
    ap.add_argument("--set-baseline", action="store_true", help="把這次的結果設為新的基準")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--embedder-url", default="http://localhost:8002")
    args = ap.parse_args()
    if args.runs < 1:
        sys.exit("--runs 至少 1")
    sys.exit(run(args))


if __name__ == "__main__":
    main()

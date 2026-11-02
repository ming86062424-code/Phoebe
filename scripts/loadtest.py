"""Stepwise load test of the chat: 1, 2, 5, then 10 users asking at the same time.

Run it only when you want numbers: every question is a real chat (it uses the chat model's
quota) and it loads the machine. See docs/loadtest.md.

    uv run python -m scripts.loadtest                      # 1, 2, 5, 10 users, 3 questions each
    uv run python -m scripts.loadtest --steps 1,3,5 --questions 2 --no-kb

It logs in as the admin account from docker/.env (never printed), creates a temporary
knowledge base from test_data/rag (unless --no-kb), and deletes everything it creates.
"""

import argparse
import json
import math
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import median

import httpx
from dotenv import dotenv_values

from scripts.benchmark import DOCKER_ENV, ROOT, TOTAL, Bench, _git_commit, _memory, _source, peaks_in_window

MAX_USERS = 10
DEFAULT_STEPS = (1, 2, 5, 10)
DATASET = ROOT / "evals" / "rag.dataset.json"
RAG_DIR = ROOT / "test_data" / "rag"
DEFAULT_OUT = ROOT / "data"
REQUEST_TIMEOUT = 300


@dataclass
class Result:
    user: int
    ok: bool
    first_byte: float | None  # seconds until the first streamed line (sources or text)
    total: float
    error: str | None = None


def parse_steps(text: str) -> list[int]:
    """'1,2,5,10' -> [1, 2, 5, 10]; each step 1..MAX_USERS, increasing."""
    try:
        steps = [int(s) for s in text.split(",") if s.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"格式應為以逗號分隔的人數，例如 1,2,5,10：{text}") from None
    if not steps or any(not 1 <= s <= MAX_USERS for s in steps):
        raise argparse.ArgumentTypeError(f"每一階的人數要在 1–{MAX_USERS} 之間：{text}")
    if steps != sorted(set(steps)):
        raise argparse.ArgumentTypeError(f"人數要由少到多、不重複：{text}")
    return steps


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (p in 0..100); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize_step(users: int, results: list[Result], wall: float) -> dict:
    ok = [r for r in results if r.ok]
    first = [r.first_byte for r in ok if r.first_byte is not None]
    total = [r.total for r in ok]
    errors: dict[str, int] = {}
    for r in results:
        if not r.ok:
            errors[r.error or "未知錯誤"] = errors.get(r.error or "未知錯誤", 0) + 1
    return {
        "users": users,
        "requests": len(results),
        "ok": len(ok),
        "error_rate": (len(results) - len(ok)) / len(results) if results else 0.0,
        "first_byte_p50": median(first) if first else None,
        "first_byte_p95": percentile(first, 95),
        "total_p50": median(total) if total else None,
        "total_p95": percentile(total, 95),
        "wall": wall,
        "per_minute": len(ok) / wall * 60 if wall > 0 else 0.0,
        "errors": errors,
    }


def _s(v: float | None) -> str:
    return "—" if v is None else f"{v:.1f} 秒"


def render_markdown(environment: dict, steps: list[dict]) -> str:
    lines = [f"# 對話壓力測試 {environment['started']}", "", "## 環境", ""]
    lines += [f"- {k}：{v}" for k, v in environment.items() if k != "started"]
    lines += ["", "## 結果", "",
              "| 同時人數 | 請求數 | 錯誤率 | 開始回應 p50／p95 | 回答完成 p50／p95 | 每分鐘完成 |",
              "|---|---|---|---|---|---|"]
    for s in steps:
        lines.append(
            f"| {s['users']} | {s['requests']} | {s['error_rate'] * 100:.0f}% | "
            f"{_s(s['first_byte_p50'])}／{_s(s['first_byte_p95'])} | "
            f"{_s(s['total_p50'])}／{_s(s['total_p95'])} | {s['per_minute']:.1f} |"
        )
    if any(s.get("peaks") for s in steps):
        sources = sorted({k for s in steps for k in s.get("peaks", {})}, key=lambda k: (k == TOTAL, k))
        lines += ["", "## 記憶體高峰", "", "| 同時人數 | " + " | ".join(_source(k) for k in sources) + " |",
                  "|---" * (len(sources) + 1) + "|"]
        for s in steps:
            peaks = s.get("peaks", {})
            lines.append(f"| {s['users']} | " + " | ".join(
                _memory(peaks[k]) if k in peaks else "—" for k in sources) + " |")
    failed = [s for s in steps if s["errors"]]
    if failed:
        lines += ["", "## 錯誤", ""]
        for s in failed:
            lines += [f"- {s['users']} 人：{msg}（{n} 次）" for msg, n in s["errors"].items()]
    return "\n".join(lines) + "\n"


# ---- running against the stack ----


def _ask(client: httpx.Client, kb_id: int | None, question: str, created: list[int],
         lock: threading.Lock, user: int) -> Result:
    started = time.perf_counter()
    first: float | None = None
    try:
        cid = client.post("/api/conversations").raise_for_status().json()["id"]
        with lock:
            created.append(cid)
        if kb_id is not None:
            client.patch(f"/api/conversations/{cid}", json={"kb_id": kb_id}).raise_for_status()
        started = time.perf_counter()  # time the message only, not the conversation setup
        last = None
        with client.stream("POST", f"/api/conversations/{cid}/messages", json={"content": question}) as r:
            if r.status_code != 200:
                r.read()
                detail = r.json().get("detail") if "json" in r.headers.get("content-type", "") else r.text
                return Result(user, False, None, time.perf_counter() - started, f"{r.status_code} {detail}")
            for line in r.iter_lines():
                if not line:
                    continue
                if first is None:
                    first = time.perf_counter() - started
                last = json.loads(line)
        total = time.perf_counter() - started
        if not last or last.get("type") != "done":
            return Result(user, False, first, total, (last or {}).get("detail") or "回答沒有正常結束")
        return Result(user, True, first, total)
    except httpx.HTTPError as e:
        return Result(user, False, first, time.perf_counter() - started, f"{type(e).__name__}: {e}")


def run_step(bench: Bench, users: int, questions: list[str], per_user: int,
             kb_id: int | None) -> tuple[list[Result], float]:
    """`users` threads at once, each asking `per_user` questions one after another."""
    results: list[Result] = []
    lock = threading.Lock()

    def worker(user: int) -> None:
        with httpx.Client(base_url=str(bench.c.base_url), cookies=bench.c.cookies,
                          timeout=REQUEST_TIMEOUT) as client:
            for i in range(per_user):
                q = questions[(user * per_user + i) % len(questions)]
                r = _ask(client, kb_id, q, bench.conversations, lock, user)
                with lock:
                    results.append(r)

    started = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(u,)) for u in range(users)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, time.perf_counter() - started


def _questions() -> list[str]:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    return [g["input"] for g in data if g["additional_metadata"]["category"] == "general"]


def run(args) -> int:
    env = dotenv_values(DOCKER_ENV)
    if not env.get("ADMIN_USERNAME") or not env.get("ADMIN_PASSWORD"):
        sys.exit("docker/.env 沒有設定 ADMIN_USERNAME／ADMIN_PASSWORD")
    bench = Bench(args.url, env["ADMIN_USERNAME"], env["ADMIN_PASSWORD"])
    questions = _questions()
    llm = bench.get("/api/settings")
    model = llm.get("gemini_model") if llm["llm_provider"] == "gemini" else llm.get("ollama_model")
    environment = {
        "started": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "git commit": _git_commit(),
        "聊天模型": f"{llm['llm_provider']} {model}",
        "知識庫": "不使用" if args.no_kb else f"{RAG_DIR.relative_to(ROOT).as_posix()} 的 md",
        "每人提問數": args.questions,
    }
    steps: list[dict] = []
    try:
        kb_id = None
        if not args.no_kb:
            files = sorted(RAG_DIR.glob("*/*.md"))
            if not files:
                sys.exit(f"{RAG_DIR} 沒有 md 檔；不需要知識庫時加上 --no-kb")
            kb_id = bench.create_kb(f"壓力測試 {time.strftime('%m%d-%H%M')}", "liteparse")
            print(f"上傳 {len(files)} 份測試文件…", flush=True)
            failed = [f.name for f in files if bench.upload(kb_id, f)["status"] != "ready"]
            if failed:
                sys.exit(f"文件處理失敗：{'、'.join(failed)}")
        interval = (bench.get("/api/monitor/status").get("collector") or {}).get("interval_seconds") or 5
        for i, users in enumerate(args.steps):
            if i:
                time.sleep(args.pause)
            print(f"{users} 人同時提問，每人 {args.questions} 題…", flush=True)
            start = time.time()
            results, wall = run_step(bench, users, questions, args.questions, kb_id)
            end = time.time()
            step = summarize_step(users, results, wall)
            time.sleep(interval + 1)  # wait for the monitor's sample after the step
            step["peaks"] = {k: round(v) for k, v in peaks_in_window(
                bench.samples(start, end, interval), start, end + interval).items()}
            steps.append(step)
            print(f"  錯誤率 {step['error_rate'] * 100:.0f}%，回答完成 p95 {_s(step['total_p95'])}", flush=True)
            if step["error_rate"] > args.stop_error_rate:
                print(f"錯誤率超過 {args.stop_error_rate * 100:.0f}%，停止加人", flush=True)
                break
    finally:
        bench.cleanup()

    report = render_markdown(environment, steps)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    md = args.out / f"loadtest-{stamp}.md"
    md.write_text(report, encoding="utf-8", newline="\n")
    md.with_suffix(".json").write_text(
        json.dumps({"environment": environment, "steps": steps}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    print("\n" + report + f"\n報告：{md}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="對話的分階段壓力測試（最多 10 人）")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--steps", type=parse_steps, default=list(DEFAULT_STEPS),
                   help=f"每一階同時提問的人數，以逗號分隔（預設 1,2,5,10，最多 {MAX_USERS}）")
    p.add_argument("--questions", type=int, default=3, help="每人在每一階依序問幾題（預設 3）")
    p.add_argument("--pause", type=float, default=10, help="每一階之間休息幾秒（預設 10）")
    p.add_argument("--stop-error-rate", type=float, default=0.2,
                   help="某一階的錯誤率超過這個比例就不再加人（預設 0.2）")
    p.add_argument("--no-kb", action="store_true", help="不建立知識庫，只測聊天模型")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="報告目錄（預設 data/，不納入 git）")
    args = p.parse_args()
    if args.questions < 1:
        p.error("--questions 至少 1")
    sys.exit(run(args))


if __name__ == "__main__":
    main()

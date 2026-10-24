"""Record container memory/CPU and GPU usage to a CSV while a test runs.

Usage (from the project root, in a second terminal):

    uv run python -m scripts.monitor_resources
    uv run python -m scripts.monitor_resources --interval 1 --out data/usage.csv --only phoebe
    uv run python -m scripts.monitor_resources --duration 600

Stop with Ctrl-C (or let --duration run out); the peaks of each container and of the GPU are printed at the end.
Needs only `docker` (and `nvidia-smi` for the GPU rows) on PATH.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

UNITS = {"b": 1 / 2**20, "kib": 1 / 2**10, "kb": 1e3 / 2**20, "mib": 1.0, "mb": 1e6 / 2**20,
         "gib": 2**10, "gb": 1e9 / 2**20, "tib": 2**20, "tb": 1e12 / 2**20}
_SIZE = re.compile(r"^\s*([0-9.]+)\s*([A-Za-z]+)\s*$")
FIELDS = ["time", "elapsed_s", "source", "mem_mib", "mem_limit_mib", "cpu_pct", "gpu_util_pct"]
TOTAL = "全部容器"
GPU = "GPU"


@dataclass(frozen=True)
class Sample:
    source: str  # container name, TOTAL or GPU
    mem_mib: float
    mem_limit_mib: float | None = None
    cpu_pct: float | None = None
    gpu_util_pct: float | None = None


def to_mib(size: str) -> float:
    """'71.79MiB' -> 71.79, '1.5GiB' -> 1536.0 (docker stats sizes)."""
    m = _SIZE.match(size)
    if not m or m.group(2).lower() not in UNITS:
        raise ValueError(f"unknown size: {size!r}")
    return float(m.group(1)) * UNITS[m.group(2).lower()]


def parse_docker_stats(lines: list[str]) -> list[Sample]:
    """`docker stats --no-stream --format '{{json .}}'` output -> one sample per container."""
    samples = []
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        used, _, limit = row["MemUsage"].partition("/")
        samples.append(
            Sample(
                source=row["Name"],
                mem_mib=to_mib(used),
                mem_limit_mib=to_mib(limit) if limit.strip() else None,
                cpu_pct=float(row["CPUPerc"].rstrip("%") or 0),
            )
        )
    return samples


def parse_nvidia_smi(output: str) -> list[Sample]:
    """`nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu
    --format=csv,noheader,nounits` -> one sample per GPU (memory in MiB)."""
    samples = []
    for i, line in enumerate(l for l in output.splitlines() if l.strip()):
        used, total, util = (part.strip() for part in line.split(","))
        name = GPU if i == 0 else f"{GPU}{i}"
        samples.append(Sample(name, float(used), float(total), gpu_util_pct=float(util)))
    return samples


def total_of(containers: list[Sample]) -> Sample:
    """All containers together, against the Docker VM's memory (the shared limit)."""
    limits = [s.mem_limit_mib for s in containers if s.mem_limit_mib]
    return Sample(
        TOTAL,
        sum(s.mem_mib for s in containers),
        max(limits) if limits else None,
        sum(s.cpu_pct or 0 for s in containers),
    )


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def collect(only: str | None) -> list[Sample]:
    samples: list[Sample] = []
    out = _run(["docker", "stats", "--no-stream", "--format", "{{json .}}"])
    if out is not None:
        containers = parse_docker_stats(out.splitlines())
        # the total covers every container: they all share the Docker VM's memory
        samples.append(total_of(containers))
        samples += [s for s in containers if not only or only in s.source]
    gpu = _run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits"])
    if gpu is not None:
        samples += parse_nvidia_smi(gpu)
    return samples


def _fmt(v: float | None) -> str:
    return "" if v is None else f"{v:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="跑測試時記錄容器與 GPU 的用量")
    ap.add_argument("--interval", type=float, default=2.0, help="取樣間隔秒數（預設 2）")
    ap.add_argument("--out", type=Path, help="CSV 路徑（預設 data/usage-<時間>.csv）")
    ap.add_argument("--only", help="只記錄名稱包含這段文字的容器，例如 phoebe（總量仍算全部）")
    ap.add_argument("--duration", type=float, help="記錄幾秒後自動停止（預設一直記錄到 Ctrl-C）")
    args = ap.parse_args()
    out = args.out or Path("data") / f"usage-{datetime.now():%Y%m%d-%H%M%S}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    peaks: dict[str, tuple[float, str]] = {}
    start = time.monotonic()
    print(f"記錄到 {out}，按 Ctrl-C 停止", flush=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        try:
            while args.duration is None or time.monotonic() - start < args.duration:
                tick = time.monotonic()
                now = datetime.now().strftime("%H:%M:%S")
                samples = collect(args.only)
                if not samples:
                    sys.exit("docker 和 nvidia-smi 都無法執行")
                for s in samples:
                    writer.writerow([now, f"{tick - start:.0f}", s.source, _fmt(s.mem_mib),
                                     _fmt(s.mem_limit_mib), _fmt(s.cpu_pct), _fmt(s.gpu_util_pct)])
                    if s.mem_mib > peaks.get(s.source, (-1.0, ""))[0]:
                        peaks[s.source] = (s.mem_mib, now)
                f.flush()
                time.sleep(max(0.0, args.interval - (time.monotonic() - tick)))
        except KeyboardInterrupt:
            pass
    print(f"\n記錄了 {time.monotonic() - start:.0f} 秒。記憶體高峰：")
    for source, (mib, at) in sorted(peaks.items(), key=lambda kv: -kv[1][0]):
        print(f"  {source:<30} {mib / 1024:6.2f} GiB  （{at}）")


if __name__ == "__main__":
    main()

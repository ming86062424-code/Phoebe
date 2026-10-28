"""Samples every container's memory and CPU (Docker Engine API) and the GPU (nvidia-smi).

Runs as the `monitor` service; docs/monitoring.md explains the Docker socket it needs.
"""

import logging
import subprocess
import threading
import time
from collections.abc import Callable

import httpx

from app.monitor import store
from app.monitor.store import Sample

logger = logging.getLogger(__name__)

DOCKER_SOCKET = "/var/run/docker.sock"
MIB = 2**20
GPU = "GPU"
PRUNE_EVERY_SECONDS = 600
NVIDIA_SMI = [
    "nvidia-smi",
    "--query-gpu=memory.used,memory.total,utilization.gpu",
    "--format=csv,noheader,nounits",
]


def memory_used(memory_stats: dict) -> float:
    """Bytes in use the way `docker stats` shows them: usage minus the inactive file cache
    (cgroup v1 `total_inactive_file`, v2 `inactive_file`)."""
    usage = memory_stats.get("usage") or 0
    stats = memory_stats.get("stats") or {}
    for key in ("total_inactive_file", "inactive_file"):
        cache = stats.get(key)
        if cache is not None and cache < usage:
            return usage - cache
    return usage


class CpuTracker:
    """CPU % from two consecutive one-shot stats: Docker's one-shot reply has no previous
    reading, so the collector keeps the last one per container (same formula as docker stats)."""

    def __init__(self) -> None:
        self._last: dict[str, tuple[int, int]] = {}

    def percent(self, container_id: str, cpu_stats: dict) -> float | None:
        total = (cpu_stats.get("cpu_usage") or {}).get("total_usage")
        system = cpu_stats.get("system_cpu_usage")
        cpus = cpu_stats.get("online_cpus") or 1
        if total is None or system is None:
            return None
        previous = self._last.get(container_id)
        self._last[container_id] = (total, system)
        if previous is None:
            return None
        d_total, d_system = total - previous[0], system - previous[1]
        if d_total < 0 or d_system <= 0:
            return None
        return d_total / d_system * cpus * 100.0

    def forget_except(self, ids: set[str]) -> None:
        for gone in set(self._last) - ids:
            del self._last[gone]


def parse_nvidia_smi(output: str) -> list[Sample]:
    """One sample per GPU (memory in MiB)."""
    samples = []
    for i, line in enumerate(l for l in output.splitlines() if l.strip()):
        used, total, util = (part.strip() for part in line.split(","))
        samples.append(
            Sample(
                source=GPU if i == 0 else f"{GPU}{i}",
                kind="gpu",
                mem_mib=float(used),
                mem_limit_mib=float(total),
                gpu_util_pct=float(util),
            )
        )
    return samples


def read_gpus(run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> list[Sample]:
    """GPU samples, or [] when there is no NVIDIA GPU (nvidia-smi missing or failing)."""
    try:
        out = run(NVIDIA_SMI, capture_output=True, text=True, check=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    try:
        return parse_nvidia_smi(out)
    except ValueError:
        logger.warning("無法解析 nvidia-smi 的輸出：%r", out)
        return []


class Collector:
    def __init__(
        self,
        app_db_path: str,
        interval: float,
        sample_hours: float,
        event_days: float,
        docker: httpx.Client | None = None,
        gpus: Callable[[], list[Sample]] = read_gpus,
        clock: Callable[[], float] = time.time,
    ):
        self.app_db_path = app_db_path
        self.interval = interval
        self.sample_hours = sample_hours
        self.event_days = event_days
        self.docker = docker or httpx.Client(
            transport=httpx.HTTPTransport(uds=DOCKER_SOCKET), base_url="http://docker", timeout=10
        )
        self.gpus = gpus
        self.clock = clock
        self.cpu = CpuTracker()
        self._last_prune = 0.0

    def containers(self) -> list[Sample]:
        listed = self.docker.get("/containers/json").raise_for_status().json()
        samples = []
        for c in listed:
            stats = self.docker.get(
                f"/containers/{c['Id']}/stats", params={"stream": "false", "one-shot": "true"}
            )
            if stats.status_code != 200:  # stopped between the two calls
                continue
            s = stats.json()
            mem = s.get("memory_stats") or {}
            labels = c.get("Labels") or {}
            samples.append(
                Sample(
                    source=(c.get("Names") or [c["Id"][:12]])[0].lstrip("/"),
                    kind="container",
                    mem_mib=memory_used(mem) / MIB,
                    mem_limit_mib=(mem["limit"] / MIB) if mem.get("limit") else None,
                    cpu_pct=self.cpu.percent(c["Id"], s.get("cpu_stats") or {}),
                    project=labels.get("com.docker.compose.project", ""),
                )
            )
        self.cpu.forget_except({c["Id"] for c in listed})
        return samples

    def sample_once(self, conn) -> None:
        now = self.clock()
        samples: list[Sample] = []
        error = ""
        try:
            samples += self.containers()
        except (httpx.HTTPError, ValueError) as e:
            error = f"無法讀取 Docker 的容器資料：{e}"
            logger.warning(error)
        samples += self.gpus()
        if samples:
            store.insert_samples(conn, now, samples)
        store.set_collector_status(conn, self.interval, now if samples else None, error)
        if now - self._last_prune >= PRUNE_EVERY_SECONDS:
            store.prune(conn, self.sample_hours, self.event_days, now)
            self._last_prune = now

    def run_forever(self, stop: threading.Event) -> None:
        conn = store.open_db(self.app_db_path)
        try:
            while not stop.is_set():
                started = time.monotonic()
                try:
                    self.sample_once(conn)
                except Exception:
                    logger.exception("資源取樣失敗")
                stop.wait(max(0.0, self.interval - (time.monotonic() - started)))
        finally:
            conn.close()

import json

import pytest

from scripts.monitor_resources import (
    GPU,
    TOTAL,
    parse_docker_stats,
    parse_nvidia_smi,
    to_mib,
    total_of,
)


def test_to_mib_reads_docker_sizes():
    assert to_mib("71.79MiB") == pytest.approx(71.79)
    assert to_mib("1.5GiB") == pytest.approx(1536)
    assert to_mib("512KiB") == pytest.approx(0.5)
    assert to_mib("0B") == 0
    assert to_mib(" 15.55GiB ") == pytest.approx(15.55 * 1024)
    with pytest.raises(ValueError):
        to_mib("lots")


def _row(name, mem, cpu):
    return json.dumps({"Name": name, "MemUsage": mem, "CPUPerc": cpu, "ID": "x"})


def test_parse_docker_stats_and_total():
    lines = [
        _row("phoebe-embedder-1", "4GiB / 15.55GiB", "3.10%"),
        "",
        _row("phoebe-web-1", "512MiB / 15.55GiB", "0.50%"),
    ]
    samples = parse_docker_stats(lines)
    assert [s.source for s in samples] == ["phoebe-embedder-1", "phoebe-web-1"]
    assert samples[0].mem_mib == pytest.approx(4096)
    assert samples[0].mem_limit_mib == pytest.approx(15.55 * 1024)
    assert samples[1].cpu_pct == pytest.approx(0.5)
    total = total_of(samples)
    assert total.source == TOTAL
    assert total.mem_mib == pytest.approx(4608)
    assert total.cpu_pct == pytest.approx(3.6)


def test_parse_nvidia_smi_one_row_per_gpu():
    samples = parse_nvidia_smi("1168, 6144, 3\n4900, 8192, 97\n")
    assert [s.source for s in samples] == [GPU, f"{GPU}1"]
    assert samples[0].mem_mib == 1168
    assert samples[0].mem_limit_mib == 6144
    assert samples[1].gpu_util_pct == 97

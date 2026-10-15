import json
import subprocess

import pytest

import law_grounding.resource_monitor as resource_monitor
from law_grounding.resource_monitor import (
    ResourceMonitor,
    ResourceLimitExceeded,
    check_resource_limits,
    parse_meminfo,
    parse_nvidia_smi_csv,
)


def test_parse_meminfo_reports_total_and_available_mib():
    text = """MemTotal:       1024000 kB
MemFree:         128000 kB
MemAvailable:    512000 kB
"""

    parsed = parse_meminfo(text)

    assert parsed["system_mem_total_mib"] == 1000.0
    assert parsed["system_mem_available_mib"] == 500.0


def test_parse_nvidia_smi_csv_reports_gpu_memory_and_utilization():
    text = "0, NVIDIA A800 80GB PCIe, 12345, 81920, 56\n1, NVIDIA A800 80GB PCIe, 0, 81920, 0\n"

    gpus = parse_nvidia_smi_csv(text)

    assert gpus == [
        {
            "index": 0,
            "name": "NVIDIA A800 80GB PCIe",
            "memory_used_mib": 12345,
            "memory_total_mib": 81920,
            "utilization_gpu_percent": 56,
        },
        {
            "index": 1,
            "name": "NVIDIA A800 80GB PCIe",
            "memory_used_mib": 0,
            "memory_total_mib": 81920,
            "utilization_gpu_percent": 0,
        },
    ]


def test_resource_monitor_writes_jsonl_with_event_and_step(tmp_path, monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="0, NVIDIA A800 80GB PCIe, 2048, 81920, 75\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    log_path = tmp_path / "resource_monitor.jsonl"
    monitor = ResourceMonitor(log_path)

    row = monitor.log("step", step=12, extra={"loss": 1.5})

    saved = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert saved["event"] == "step"
    assert saved["step"] == 12
    assert saved["loss"] == 1.5
    assert saved["gpus"][0]["memory_used_mib"] == 2048
    assert row == saved


def test_check_resource_limits_detects_system_memory_and_gpu_thresholds():
    row = {
        "system_mem_total_mib": 1000.0,
        "system_mem_available_mib": 150.0,
        "gpus": [
            {"index": 0, "memory_used_mib": 700, "memory_total_mib": 1000},
            {"index": 1, "memory_used_mib": 850, "memory_total_mib": 1000},
        ],
    }

    violations = check_resource_limits(row, max_system_mem_percent=80.0, max_gpu_mem_percent=80.0)

    assert "system_mem_percent=85.00>=80.00" in violations
    assert "gpu1_mem_percent=85.00>=80.00" in violations
    assert all("gpu0" not in item for item in violations)


def test_resource_monitor_raises_after_writing_limit_exceeded_row(tmp_path, monkeypatch):
    monkeypatch.setattr(
        resource_monitor,
        "collect_resource_snapshot",
        lambda: {
            "timestamp": "2026-06-26T00:00:00+0800",
            "pid": 123,
            "hostname": "test-host",
            "system_mem_total_mib": 1000.0,
            "system_mem_available_mib": 100.0,
            "gpus": [{"index": 0, "memory_used_mib": 900, "memory_total_mib": 1000}],
        },
    )
    log_path = tmp_path / "resource_monitor.jsonl"
    monitor = ResourceMonitor(
        log_path,
        max_system_mem_percent=80.0,
        max_gpu_mem_percent=80.0,
        print_to_stdout=False,
    )

    with pytest.raises(ResourceLimitExceeded) as exc:
        monitor.log("train_step", step=5)

    assert "system_mem_percent=90.00>=80.00" in str(exc.value)
    assert "gpu0_mem_percent=90.00>=80.00" in str(exc.value)
    saved = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert saved["limit_exceeded"] is True
    assert saved["limit_violations"] == [
        "system_mem_percent=90.00>=80.00",
        "gpu0_mem_percent=90.00>=80.00",
    ]

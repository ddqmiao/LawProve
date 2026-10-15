"""Lightweight GPU and memory monitoring for long-running experiments."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


class ResourceLimitExceeded(RuntimeError):
    """Raised when configured memory limits are exceeded."""


def _kib_to_mib(value_kib: int | float) -> float:
    return round(float(value_kib) / 1024.0, 3)


def parse_meminfo(text: str) -> dict[str, float]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if parts and parts[0].isdigit():
            values[key] = int(parts[0])
    result: dict[str, float] = {}
    if "MemTotal" in values:
        result["system_mem_total_mib"] = _kib_to_mib(values["MemTotal"])
    if "MemAvailable" in values:
        result["system_mem_available_mib"] = _kib_to_mib(values["MemAvailable"])
    return result


def read_system_memory() -> dict[str, float]:
    try:
        return parse_meminfo(Path("/proc/meminfo").read_text(encoding="utf-8"))
    except OSError:
        return {}


def read_process_rss_mib() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return _kib_to_mib(int(parts[1]))
    except OSError:
        return None
    return None


def parse_nvidia_smi_csv(text: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mib": int(parts[2]),
                    "memory_total_mib": int(parts[3]),
                    "utilization_gpu_percent": int(parts[4]),
                }
            )
        except ValueError:
            continue
    return gpus


def query_nvidia_smi(timeout_sec: float = 5.0) -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return parse_nvidia_smi_csv(result.stdout)


def torch_cuda_memory() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {}
    if not torch.cuda.is_available():
        return {"torch_cuda_available": False}
    device = torch.cuda.current_device()
    return {
        "torch_cuda_available": True,
        "torch_cuda_device": device,
        "torch_cuda_device_name": torch.cuda.get_device_name(device),
        "torch_memory_allocated_mib": round(torch.cuda.memory_allocated(device) / 1024**2, 3),
        "torch_memory_reserved_mib": round(torch.cuda.memory_reserved(device) / 1024**2, 3),
        "torch_max_memory_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 3),
        "torch_max_memory_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 3),
    }


def collect_resource_snapshot() -> dict[str, Any]:
    row: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
    }
    row.update(read_system_memory())
    rss = read_process_rss_mib()
    if rss is not None:
        row["process_rss_mib"] = rss
    row["gpus"] = query_nvidia_smi()
    row.update(torch_cuda_memory())
    return row


def check_resource_limits(
    row: dict[str, Any],
    *,
    max_system_mem_percent: float = 80.0,
    max_gpu_mem_percent: float = 80.0,
    gpu_indices: set[int] | None = None,
) -> list[str]:
    """Return resource limit violation messages for a snapshot.

    A threshold <= 0 disables that limit.
    """
    violations: list[str] = []
    if max_system_mem_percent > 0:
        total = row.get("system_mem_total_mib")
        available = row.get("system_mem_available_mib")
        if total and available is not None:
            used_percent = (float(total) - float(available)) / float(total) * 100.0
            if used_percent >= max_system_mem_percent:
                violations.append(f"system_mem_percent={used_percent:.2f}>={max_system_mem_percent:.2f}")

    if max_gpu_mem_percent > 0:
        for gpu in row.get("gpus") or []:
            try:
                index = int(gpu["index"])
                used = float(gpu["memory_used_mib"])
                total = float(gpu["memory_total_mib"])
            except (KeyError, TypeError, ValueError):
                continue
            if gpu_indices is not None and index not in gpu_indices:
                continue
            if total <= 0:
                continue
            used_percent = used / total * 100.0
            if used_percent >= max_gpu_mem_percent:
                violations.append(f"gpu{index}_mem_percent={used_percent:.2f}>={max_gpu_mem_percent:.2f}")
    return violations


class ResourceMonitor:
    """Append resource snapshots to JSONL and optionally print compact lines."""

    def __init__(
        self,
        log_path: str | Path,
        *,
        print_to_stdout: bool = True,
        max_system_mem_percent: float = 80.0,
        max_gpu_mem_percent: float = 80.0,
        gpu_indices: set[int] | None = None,
    ):
        self.log_path = Path(log_path)
        self.print_to_stdout = print_to_stdout
        self.max_system_mem_percent = max_system_mem_percent
        self.max_gpu_mem_percent = max_gpu_mem_percent
        self.gpu_indices = gpu_indices
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, *, step: int | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        row = collect_resource_snapshot()
        row["event"] = event
        if step is not None:
            row["step"] = step
        if extra:
            row.update(extra)
        violations = check_resource_limits(
            row,
            max_system_mem_percent=self.max_system_mem_percent,
            max_gpu_mem_percent=self.max_gpu_mem_percent,
            gpu_indices=self.gpu_indices,
        )
        if violations:
            row["limit_exceeded"] = True
            row["limit_violations"] = violations
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        if self.print_to_stdout:
            print(format_resource_summary(row), flush=True)
        if violations:
            raise ResourceLimitExceeded("; ".join(violations))
        return row


def format_resource_summary(row: dict[str, Any]) -> str:
    parts = [f"[resource] event={row.get('event')}"]
    if "step" in row:
        parts.append(f"step={row['step']}")
    if "process_rss_mib" in row:
        parts.append(f"rss={row['process_rss_mib']:.0f}MiB")
    if "system_mem_available_mib" in row:
        parts.append(f"mem_avail={row['system_mem_available_mib']:.0f}MiB")
    gpus = row.get("gpus") or []
    if gpus:
        gpu_parts = [
            f"{gpu['index']}:{gpu['memory_used_mib']}/{gpu['memory_total_mib']}MiB,{gpu['utilization_gpu_percent']}%"
            for gpu in gpus
        ]
        parts.append("gpu=" + ";".join(gpu_parts))
    if "torch_memory_reserved_mib" in row:
        parts.append(f"torch_reserved={row['torch_memory_reserved_mib']:.0f}MiB")
    if row.get("limit_exceeded"):
        parts.append("LIMIT_EXCEEDED=" + "|".join(row.get("limit_violations") or []))
    return " ".join(parts)

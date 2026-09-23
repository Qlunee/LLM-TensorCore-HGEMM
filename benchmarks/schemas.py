"""Shared V0 benchmark schema, statistics, and environment manifest."""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch


CSV_FIELDS = [
    "timestamp", "git_sha", "gpu", "sm", "driver", "cuda", "torch",
    "provider", "version", "M", "N", "K", "layout", "input_dtype",
    "accum_dtype", "output_dtype", "algo_id", "workspace_bytes", "warmup",
    "samples", "launches_per_sample", "median_us", "p95_us", "min_us",
    "tflops", "ratio_to_cublaslt", "max_abs_error", "mean_abs_error",
    "max_rel_error", "relative_l2_error", "nan_count", "inf_count",
    "registers_per_thread", "smem_per_block", "achieved_occupancy",
    "dram_pct", "l2_pct", "tensor_pipe_metric", "status",
]


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup: int = 20
    samples: int = 50
    launches_per_sample: int = 10


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_us(values: list[float]) -> dict[str, float]:
    return {
        "median_us": statistics.median(values),
        "p95_us": percentile(values, 0.95),
        "min_us": min(values),
    }


def _command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def collect_environment() -> dict[str, object]:
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    sm = f"{properties.major}.{properties.minor}"
    smi = _command(
        "nvidia-smi",
        f"--id={device}",
        "--query-gpu=name,driver_version,clocks.sm,clocks.mem,power.limit,temperature.gpu,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda or "unknown",
        "cuda_runtime": torch._C._cuda_getCompiledVersion(),
        "gpu": properties.name,
        "sm": sm,
        "nvidia_smi": smi,
        "nvcc": _command("nvcc", "--version"),
        "git_sha": _command("git", "rev-parse", "HEAD"),
        "compile_arch": "sm_86",
        "contract": {
            "input_dtype": "fp16",
            "accumulation_dtype": "fp32",
            "output_dtype": "fp16",
            "layout": "row-major/row-major",
            "alpha": 1.0,
            "beta": 0.0,
            "allow_tf32": False,
        },
    }


def save_manifest(path: Path, environment: dict[str, object], config: BenchmarkConfig) -> None:
    payload = {"environment": environment, "benchmark": asdict(config)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

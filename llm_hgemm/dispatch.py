"""Exact-shape dispatch with production and custom-only policies."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import torch

from .provenance import code_fingerprint, file_sha256


CUSTOM_PROVIDERS = (
    "wmma_tiled",
    "mma_ptx",
    "mma_vectorized",
    "mma_async",
    "mma_async_compact",
)

STRATEGIES = ("best_available", "best_custom")
_ALLOWED = set(CUSTOM_PROVIDERS) | {"cublaslt"}

_policy: dict | None = None
_local = threading.local()


def configure_dispatch(
    path: str | Path | None,
    *,
    strategy: str = "best_available",
) -> None:
    """Configure outside the inference hot path."""
    global _policy

    if strategy not in STRATEGIES:
        raise ValueError(f"unknown dispatch strategy: {strategy}")

    if path is None:
        _policy = None
        return

    payload = json.loads(Path(path).read_text())
    if payload["schema_version"] != 2:
        raise ValueError("V7 requires dispatch schema_version=2")

    if payload["torch"] != torch.__version__:
        raise ValueError("dispatch table PyTorch version mismatch")
    if payload["cuda"] != torch.version.cuda:
        raise ValueError("dispatch table CUDA version mismatch")

    if payload.get("code_fingerprint") != code_fingerprint():
        raise ValueError("dispatch table source/extension fingerprint mismatch")
    if str(payload.get("sm")) != "8.6":
        raise ValueError("dispatch table must target SM86")
    entries = {}
    for item in payload["entries"]:
        key = (int(item["M"]), int(item["N"]), int(item["K"]))
        if min(key) <= 0 or key in entries:
            raise ValueError(f"invalid/duplicate shape: {key}")

        for name in STRATEGIES:
            if item[name] not in _ALLOWED:
                raise ValueError(f"invalid provider: {item[name]}")

        entries[key] = item[strategy]

    eligible_devices = {
        index
        for index in range(torch.cuda.device_count())
        if torch.cuda.get_device_name(index) == payload["gpu"]
        and torch.cuda.get_device_capability(index) == (8, 6)
    }

    _policy = {
        "strategy": strategy,
        "table_sha256": file_sha256(path),
        "entries": entries,
        "eligible_devices": eligible_devices,
    }


def selected_implementation(
    a: torch.Tensor,
    b: torch.Tensor,
    requested: str = "shape_auto",
) -> str:
    if requested != "shape_auto":
        return requested

    valid_contract = (
        a.is_cuda
        and b.is_cuda
        and a.device == b.device
        and a.dtype == torch.float16
        and b.dtype == torch.float16
        and a.ndim == 2
        and b.ndim == 2
        and a.is_contiguous()
        and b.is_contiguous()
        and a.shape[1] == b.shape[0]
        and min(a.shape[0], a.shape[1], b.shape[1]) > 0
    )
    if not valid_contract:
        # Existing C++ validation will reject invalid inputs.
        return "cublaslt"

    policy = _policy
    if policy is None or a.device.index not in policy["eligible_devices"]:
        return "cublaslt"

    # Only use our compiled SM86 fallback on an eligible device.
    if a.data_ptr() % 16 or b.data_ptr() % 16:
        return "mma_vectorized"
    key = (a.shape[0], b.shape[1], a.shape[1])
    return policy["entries"].get(key, "cublaslt")


def record_selected(provider: str) -> None:
    _local.provider = provider


def last_selected() -> str:
    provider = getattr(_local, "provider", None)
    if provider is None:
        raise RuntimeError("shape_auto has not executed in this thread")
    return provider

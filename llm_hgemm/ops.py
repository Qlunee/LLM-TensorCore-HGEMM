"""Stable Python interface for all HGEMM implementations."""

from __future__ import annotations

import torch

try:
    from . import _C
except ImportError as exc:  # pragma: no cover - exercised before installation only
    raise ImportError(
        "llm_hgemm._C is not built. Install the project with "
        "`python -m pip install -v -e .` first."
    ) from exc


def hgemm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    implementation: str = "auto",
    epilogue: str = "none",
) -> torch.Tensor:
    """Compute row-major FP16 C=A@B with FP32 accumulation."""
    return _C.hgemm(a, b, implementation, epilogue)


def _hgemm_out(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    *,
    implementation: str,
    epilogue: str = "none",
) -> torch.Tensor:
    """Benchmark seam using a preallocated output tensor."""
    _C.hgemm_out(a, b, out, implementation, epilogue)
    return out


def available_providers() -> tuple[str, ...]:
    providers = [
        "torch",
        "cuda_naive",
        "cuda_tiled",
        "wmma_basic",
        "wmma_tiled",
        "cublas",
        "cublaslt",
    ]
    if _C.has_cutlass():
        providers.append("cutlass")
    return tuple(providers)


def backend_info(implementation: str) -> dict[str, int]:
    if implementation == "torch":
        return {"algorithm_id": -1, "workspace_bytes": 0}
    return dict(_C.backend_info(implementation))

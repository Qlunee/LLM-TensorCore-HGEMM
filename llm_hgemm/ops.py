"""Stable Python interface for all HGEMM implementations."""

from __future__ import annotations

import torch

from .dispatch import last_selected, record_selected, selected_implementation

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
    resolved = selected_implementation(a, b, implementation)
    output = _C.hgemm(a, b, resolved, epilogue)
    if implementation == "shape_auto":
        record_selected(resolved)
    return output


def _hgemm_out(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    *,
    implementation: str,
    epilogue: str = "none",
) -> torch.Tensor:
    """Benchmark seam using a preallocated output tensor."""
    resolved = selected_implementation(a, b, implementation)
    _C.hgemm_out(a, b, out, resolved, epilogue)
    if implementation == "shape_auto":
        record_selected(resolved)
    return out


def available_providers() -> tuple[str, ...]:
    providers = [
        "torch",
        "shape_auto",
        "cuda_naive",
        "cuda_tiled",
        "wmma_basic",
        "wmma_tiled",
        "mma_ptx",
        "mma_padded",
        "mma_vectorized",
        "mma_double_buffer",
        "mma_async",
        "mma_double_buffer_compact",
        "mma_async_compact",
        "cublas",
        "cublaslt",
    ]
    if _C.has_cutlass():
        providers.append("cutlass")
    return tuple(providers)


def backend_info(implementation: str) -> dict[str, int]:
    if implementation == "torch":
        return {"algorithm_id": -1, "workspace_bytes": 0}
    if implementation == "shape_auto":
        implementation = last_selected()
    return dict(_C.backend_info(implementation))

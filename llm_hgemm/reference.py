"""Numerical reference and correctness gates for HGEMM implementations."""

from __future__ import annotations

from contextlib import contextmanager

import torch

MAX_ABS_ERROR_LIMIT = 5e-2
RELATIVE_L2_ERROR_LIMIT = 5e-3


@contextmanager
def _tf32_disabled():
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@torch.no_grad()
def torch_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """FP32 matmul followed by the contract's FP16 output rounding."""
    with _tf32_disabled():
        return torch.mm(a.float(), b.float()).half()


@torch.no_grad()
def error_metrics(out: torch.Tensor, reference: torch.Tensor) -> dict[str, float | int]:
    out32 = out.float()
    ref32 = reference.float()
    difference = out32 - ref32
    absolute = difference.abs()
    denominator = ref32.abs().clamp_min(1e-6)
    ref_norm = torch.linalg.vector_norm(ref32).clamp_min(1e-12)
    return {
        "max_abs_error": float(absolute.max().item()),
        "mean_abs_error": float(absolute.mean().item()),
        "max_rel_error": float((absolute / denominator).max().item()),
        "relative_l2_error": float(
            (torch.linalg.vector_norm(difference) / ref_norm).item()
        ),
        "nan_count": int(torch.isnan(out32).sum().item()),
        "inf_count": int(torch.isinf(out32).sum().item()),
    }


def correctness_passed(metrics: dict[str, float | int]) -> bool:
    return (
        metrics["nan_count"] == 0
        and metrics["inf_count"] == 0
        and metrics["max_abs_error"] <= MAX_ABS_ERROR_LIMIT
        and metrics["relative_l2_error"] <= RELATIVE_L2_ERROR_LIMIT
    )

import pytest
import torch

from llm_hgemm import available_providers, hgemm
from llm_hgemm.reference import correctness_passed, error_metrics, torch_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "provider",
    [
        "cuda_naive",
        "cuda_tiled",
        "wmma_basic",
        "wmma_tiled",
        "mma_ptx",
        "mma_vectorized",
        "cublas",
        "cublaslt",
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (16, 16, 16),
        (32, 64, 48),
        (37, 53, 29),
        (128, 256, 64),
        (129, 77, 1003),
    ],
)
def test_hgemm_backends(provider, shape):
    torch.manual_seed(2026)
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.1
    b = torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.1
    out = hgemm(a, b, implementation=provider)
    metrics = error_metrics(out, torch_reference(a, b))
    assert correctness_passed(metrics), metrics


@pytest.mark.skipif("cutlass" not in available_providers(), reason="CUTLASS not built")
def test_cutlass_aligned_shape():
    torch.manual_seed(2026)
    a = torch.randn((128, 64), device="cuda", dtype=torch.float16) * 0.1
    b = torch.randn((64, 128), device="cuda", dtype=torch.float16) * 0.1
    out = hgemm(a, b, implementation="cutlass")
    metrics = error_metrics(out, torch_reference(a, b))
    assert correctness_passed(metrics), metrics


@pytest.mark.parametrize(
    "provider",
    ["cuda_naive", "cuda_tiled", "wmma_basic", "wmma_tiled", "mma_ptx",
     "mma_vectorized"],
)
@pytest.mark.parametrize("pattern", ["zeros", "uniform", "sparse"])
def test_custom_kernel_input_patterns(provider, pattern):
    torch.manual_seed(2026)
    if pattern == "zeros":
        a = torch.zeros((33, 47), device="cuda", dtype=torch.float16)
        b = torch.zeros((47, 29), device="cuda", dtype=torch.float16)
    elif pattern == "uniform":
        a = (torch.rand((33, 47), device="cuda", dtype=torch.float16) - 0.5) * 0.2
        b = (torch.rand((47, 29), device="cuda", dtype=torch.float16) - 0.5) * 0.2
    else:
        a = torch.randn((33, 47), device="cuda", dtype=torch.float16) * 0.1
        b = torch.randn((47, 29), device="cuda", dtype=torch.float16) * 0.1
        a[a.abs() < 0.12] = 0
        b[b.abs() < 0.12] = 0
    out = hgemm(a, b, implementation=provider)
    metrics = error_metrics(out, torch_reference(a, b))
    assert correctness_passed(metrics), metrics

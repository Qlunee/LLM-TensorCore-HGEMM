import pytest
import torch

from llm_hgemm import hgemm
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
        "mma_async",
        "cublas",
        "cublaslt",
    ],
)
def test_non_default_stream(provider):
    torch.manual_seed(2026)
    # WMMA-aligned shape: V2 exercises its fast path, while V3 exercises its
    # predicate-enabled CTA tail path because N is not a multiple of 128.
    a = torch.randn((128, 64), device="cuda", dtype=torch.float16) * 0.1
    b = torch.randn((64, 80), device="cuda", dtype=torch.float16) * 0.1
    reference = torch_reference(a, b)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out = hgemm(a, b, implementation=provider)
    torch.cuda.current_stream().wait_stream(stream)
    metrics = error_metrics(out, reference)
    assert correctness_passed(metrics), metrics

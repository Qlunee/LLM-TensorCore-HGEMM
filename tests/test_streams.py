import pytest
import torch

from llm_hgemm import hgemm
from llm_hgemm.reference import correctness_passed, error_metrics, torch_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "provider", ["cuda_naive", "cuda_tiled", "wmma_basic", "cublas", "cublaslt"]
)
def test_non_default_stream(provider):
    torch.manual_seed(2026)
    # Fully aligned so wmma_basic exercises its Tensor Core fast path rather
    # than the V1 tiled fallback.
    a = torch.randn((128, 64), device="cuda", dtype=torch.float16) * 0.1
    b = torch.randn((64, 80), device="cuda", dtype=torch.float16) * 0.1
    reference = torch_reference(a, b)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out = hgemm(a, b, implementation=provider)
    torch.cuda.current_stream().wait_stream(stream)
    metrics = error_metrics(out, reference)
    assert correctness_passed(metrics), metrics

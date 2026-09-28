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
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1),
        (15, 17, 31),
        (17, 33, 65),
        (127, 129, 63),
        (129, 257, 33),
    ],
)
def test_non_tile_aligned_shapes(provider, shape):
    torch.manual_seed(7)
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.1
    b = torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.1
    out = hgemm(a, b, implementation=provider)
    metrics = error_metrics(out, torch_reference(a, b))
    assert correctness_passed(metrics), metrics

import pytest
import torch

from llm_hgemm import hgemm
from llm_hgemm.reference import correctness_passed, error_metrics, torch_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("provider", ["mma_ptx_probe", "ldmatrix_probe"])
@pytest.mark.parametrize(
    "shape",
    [
        (16, 8, 16),
        (32, 16, 32),
        (17, 9, 19),
        (37, 21, 47),
    ],
)
def test_v4_probe_mapping(provider, shape):
    torch.manual_seed(2026)
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.1
    b = torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.1
    out = hgemm(a, b, implementation=provider)
    metrics = error_metrics(out, torch_reference(a, b))
    assert correctness_passed(metrics), metrics

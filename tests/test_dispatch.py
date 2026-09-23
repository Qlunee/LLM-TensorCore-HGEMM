import pytest
import torch

from llm_hgemm import hgemm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_auto_matches_cublaslt():
    torch.manual_seed(2026)
    a = torch.randn((64, 48), device="cuda", dtype=torch.float16)
    b = torch.randn((48, 80), device="cuda", dtype=torch.float16)
    auto = hgemm(a, b, implementation="auto")
    explicit = hgemm(a, b, implementation="cublaslt")
    torch.testing.assert_close(auto, explicit, rtol=0, atol=0)

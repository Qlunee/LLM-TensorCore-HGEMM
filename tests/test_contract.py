import pytest
import torch

from llm_hgemm import hgemm
from llm_hgemm.ops import _hgemm_out

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def tensors(m=16, n=16, k=16):
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    return a, b


def test_rejects_cpu_tensor():
    a = torch.randn((16, 16), dtype=torch.float16)
    b = torch.randn((16, 16), dtype=torch.float16)
    with pytest.raises(RuntimeError, match="CUDA"):
        hgemm(a, b, implementation="cublas")


def test_rejects_wrong_dtype():
    a = torch.randn((16, 16), device="cuda", dtype=torch.float32)
    b = torch.randn((16, 16), device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="float16"):
        hgemm(a, b, implementation="cublas")


def test_rejects_k_mismatch():
    a, b = tensors(m=8, n=9, k=7)
    b = torch.randn((6, 9), device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="K dimensions"):
        hgemm(a, b, implementation="cublas")


def test_rejects_noncontiguous_input():
    a, b = tensors()
    with pytest.raises(RuntimeError, match="contiguous"):
        hgemm(a.t(), b, implementation="cublas")


def test_rejects_zero_dimension():
    a = torch.empty((0, 16), device="cuda", dtype=torch.float16)
    b = torch.empty((16, 8), device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="positive"):
        hgemm(a, b, implementation="cublas")


def test_rejects_bad_output():
    a, b = tensors(m=8, n=9, k=7)
    out = torch.empty((8, 8), device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="out must have shape"):
        _hgemm_out(a, b, out, implementation="cublas")


def test_rejects_unknown_implementation():
    a, b = tensors()
    with pytest.raises(RuntimeError, match="unknown implementation"):
        hgemm(a, b, implementation="not-a-kernel")


def test_rejects_non_none_epilogue():
    a, b = tensors()
    with pytest.raises(RuntimeError, match="epilogue"):
        hgemm(a, b, implementation="cublas", epilogue="bias")

import pytest
import torch

from llm_hgemm import available_providers, hgemm
from llm_hgemm.ops import _hgemm_out, backend_info
from llm_hgemm.reference import correctness_passed, error_metrics, torch_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

PROVIDERS = ["mma_padded", "mma_vectorized"]


def assert_correct(out, reference):
    metrics = error_metrics(out, reference)
    assert correctness_passed(metrics), metrics


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize(
    "shape",
    [
        (128, 128, 32),
        (256, 256, 64),
        (384, 1536, 768),
        (127, 129, 63),
        (129, 257, 33),
        (128, 128, 33),  # K tail only
        (128, 129, 32),  # N tail only
    ],
)
def test_v5_paths(provider, shape):
    torch.manual_seed(2026)
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.1
    b = torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.1
    out = hgemm(a, b, implementation=provider)
    assert_correct(out, torch_reference(a, b))


@pytest.mark.parametrize("offset_a,offset_b", [(1, 0), (0, 1), (1, 1)])
def test_misaligned_storage_scalar_fallback(offset_a, offset_b):
    torch.manual_seed(2026)
    m, n, k = 128, 128, 32
    a_storage = torch.randn(m * k + offset_a, device="cuda", dtype=torch.float16) * 0.1
    b_storage = torch.randn(k * n + offset_b, device="cuda", dtype=torch.float16) * 0.1
    a = a_storage[offset_a:].view(m, k)
    b = b_storage[offset_b:].view(k, n)
    assert a.is_contiguous() and b.is_contiguous()
    assert a.data_ptr() % 16 == offset_a * 2
    assert b.data_ptr() % 16 == offset_b * 2
    out = hgemm(a, b, implementation="mma_vectorized")
    assert_correct(out, torch_reference(a, b))


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("shape", [(128, 128, 32), (127, 129, 63)])
def test_v5_stream_and_preallocated_output(provider, shape):
    torch.manual_seed(2026)
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.1
    b = torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.1
    reference = torch_reference(a, b)
    out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.float16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        returned = _hgemm_out(a, b, out, implementation=provider)
    torch.cuda.current_stream().wait_stream(stream)
    assert returned is out
    assert_correct(out, reference)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_v5_provider_metadata(provider):
    assert provider in available_providers()
    assert backend_info(provider) == {"algorithm_id": -1, "workspace_bytes": 0}

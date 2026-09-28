import pytest
import torch

from llm_hgemm import available_providers, hgemm
from llm_hgemm.ops import _hgemm_out, backend_info
from llm_hgemm.reference import correctness_passed, error_metrics, torch_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

PROVIDERS = [
    "mma_double_buffer", "mma_async",
    "mma_double_buffer_compact", "mma_async_compact",
]


def assert_correct(output, reference):
    metrics = error_metrics(output, reference)
    assert correctness_passed(metrics), metrics


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize(
    "shape",
    [
        (64, 64, 32),  # Compact full tile; original boundary tile
        (128, 128, 32),  # Prologue and final computation only
        (128, 128, 64),  # First stage switch
        (128, 128, 96),  # First reuse of Stage 0
        (128, 128, 160),  # Repeated stage reuse
        (256, 256, 512),  # Multiple CTAs and many stages
        (127, 136, 40),  # Async M/N/K tails and zero-fill
        (129, 264, 72),  # Multiple boundary CTAs
        (1, 8, 8),  # Almost completely zero-filled CTA
        (127, 129, 63),  # Scalar fallback for non-vector strides
        (129, 257, 33),  # Scalar fallback for non-vector strides
    ],
)
def test_v6_pipeline_shapes(provider, shape):
    torch.manual_seed(2026)
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.1
    b = torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.1
    reference = torch_reference(a, b)
    output = hgemm(a, b, implementation=provider)
    assert_correct(output, reference)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("offset_a,offset_b", [(1, 0), (0, 1), (1, 1)])
def test_v6_misaligned_pointer_fallback(provider, offset_a, offset_b):
    torch.manual_seed(2026)
    m, n, k = 128, 128, 64
    a_storage = torch.randn(m * k + offset_a, device="cuda", dtype=torch.float16) * 0.1
    b_storage = torch.randn(k * n + offset_b, device="cuda", dtype=torch.float16) * 0.1
    a = a_storage[offset_a:].view(m, k)
    b = b_storage[offset_b:].view(k, n)
    assert a.is_contiguous() and b.is_contiguous()
    assert a.data_ptr() % 16 == offset_a * 2
    assert b.data_ptr() % 16 == offset_b * 2
    output = hgemm(a, b, implementation=provider)
    assert_correct(output, torch_reference(a, b))


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("shape", [(128, 128, 96), (127, 136, 40), (127, 129, 63)])
def test_v6_non_default_stream(provider, shape):
    torch.manual_seed(2026)
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.1
    b = torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.1
    reference = torch_reference(a, b)
    output = torch.full((m, n), float("nan"), device="cuda", dtype=torch.float16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        returned = _hgemm_out(a, b, output, implementation=provider)
    torch.cuda.current_stream().wait_stream(stream)
    assert returned is output
    assert_correct(output, reference)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_v6_provider_metadata(provider):
    assert provider in available_providers()
    assert backend_info(provider) == {"algorithm_id": -1, "workspace_bytes": 0}


@pytest.mark.parametrize("provider", PROVIDERS)
def test_v6_stage_markers(provider):
    """Distinct K-tile markers expose missing, repeated or stale stages."""
    m, n, k = 129, 136, 160
    a = torch.full((m, k), 0.125, device="cuda", dtype=torch.float16)
    b = torch.empty((k, n), device="cuda", dtype=torch.float16)
    for tile, marker in enumerate([1, 2, 4, 8, 16]):
        b[tile * 32:(tile + 1) * 32].fill_(marker / 16.0)
    output = hgemm(a, b, implementation=provider)
    # 32 * (1/8) * (1+2+4+8+16)/16 = 7.75, exactly representable.
    expected = torch.full((m, n), 7.75, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)

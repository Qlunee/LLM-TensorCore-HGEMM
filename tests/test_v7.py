import json
import pytest
import torch

from llm_hgemm import hgemm
from llm_hgemm.dispatch import (
    configure_dispatch, selected_implementation, last_selected,
)
from llm_hgemm.integration import FrozenHgemmLinear
from llm_hgemm.ops import _hgemm_out, backend_info
from llm_hgemm.provenance import code_fingerprint
from llm_hgemm.reference import correctness_passed, error_metrics, torch_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

@pytest.fixture(autouse=True)
def reset_policy():
    configure_dispatch(None)
    yield
    configure_dispatch(None)

def policy_file(tmp_path):
    path = tmp_path / "dispatch.json"
    path.write_text(json.dumps({
        "schema_version": 2, "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda, "torch": torch.__version__, "sm": "8.6",
        "code_fingerprint": code_fingerprint(),
        "entries": [{"M": 128, "N": 128, "K": 64,
                     "best_available": "cublaslt",
                     "best_custom": "mma_async_compact"}],
    }))
    return path

def tensors(m=128, n=128, k=64):
    torch.manual_seed(2026)
    return (torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.1,
            torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.1)

@pytest.mark.parametrize("strategy,expected", [
    ("best_available", "cublaslt"), ("best_custom", "mma_async_compact")])
def test_strategy_selection_and_correctness(tmp_path, strategy, expected):
    if torch.cuda.get_device_capability() != (8, 6):
        pytest.skip("SM86 required")
    configure_dispatch(policy_file(tmp_path), strategy=strategy)
    a, b = tensors()
    assert selected_implementation(a, b) == expected
    out = hgemm(a, b, implementation="shape_auto")
    assert last_selected() == expected
    assert correctness_passed(error_metrics(out, torch_reference(a, b)))
    assert backend_info("shape_auto") == backend_info(expected)

@pytest.mark.parametrize("strategy", ["best_available", "best_custom"])
@pytest.mark.parametrize("shape", [(127, 129, 63), (1, 136, 40), (65, 80, 48)])
def test_unknown_shape(tmp_path, strategy, shape):
    configure_dispatch(policy_file(tmp_path), strategy=strategy)
    a, b = tensors(*shape)
    assert selected_implementation(a, b) == "cublaslt"
    out = hgemm(a, b, implementation="shape_auto")
    assert correctness_passed(error_metrics(out, torch_reference(a, b)))

def test_misaligned_fallback(tmp_path):
    if torch.cuda.get_device_capability() != (8, 6):
        pytest.skip("SM86 required")
    configure_dispatch(policy_file(tmp_path), strategy="best_custom")
    a, b = tensors()
    storage = torch.empty(a.numel() + 1, device="cuda", dtype=torch.float16)
    offset = storage[1:].view_as(a)
    offset.copy_(a)
    assert selected_implementation(offset, b) == "mma_vectorized"
    out = hgemm(offset, b, implementation="shape_auto")
    assert correctness_passed(error_metrics(out, torch_reference(offset, b)))

def test_nondefault_stream(tmp_path):
    configure_dispatch(policy_file(tmp_path), strategy="best_custom")
    a, b = tensors()
    reference = torch_reference(a, b)
    out = torch.full((128, 128), float("nan"), device="cuda", dtype=torch.float16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        returned = _hgemm_out(a, b, out, implementation="shape_auto")
    torch.cuda.current_stream().wait_stream(stream)
    assert returned is out
    assert correctness_passed(error_metrics(out, reference))

@pytest.mark.parametrize("shape", [(2, 64, 64), (3, 64), (64,)])
def test_adapter(tmp_path, shape):
    configure_dispatch(policy_file(tmp_path), strategy="best_custom")
    layer = torch.nn.Linear(64, 128, bias=False, device="cuda",
                           dtype=torch.float16).eval()
    adapter = FrozenHgemmLinear(layer)
    x = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.1
    with torch.inference_mode():
        actual, expected = adapter(x), layer(x)
    assert actual.shape == expected.shape
    assert correctness_passed(error_metrics(actual, expected))

def test_adapter_noncontiguous_and_empty():
    layer = torch.nn.Linear(64, 128, bias=False, device="cuda",
                           dtype=torch.float16).eval()
    adapter = FrozenHgemmLinear(layer)
    inputs = [
        torch.randn((64, 3), device="cuda", dtype=torch.float16).T,
        torch.empty((0, 64), device="cuda", dtype=torch.float16),
    ]
    with torch.inference_mode():
        for x in inputs:
            torch.testing.assert_close(adapter(x), layer(x), rtol=0, atol=0)

def test_training_rejected():
    layer = torch.nn.Linear(64, 128, bias=False, device="cuda", dtype=torch.float16)
    adapter = FrozenHgemmLinear(layer)
    with pytest.raises(RuntimeError, match="inference-only"):
        adapter(torch.randn((2, 64), device="cuda", dtype=torch.float16))

def test_changed_weight_rejected():
    layer = torch.nn.Linear(64, 128, bias=False, device="cuda", dtype=torch.float16)
    adapter = FrozenHgemmLinear(layer)
    with torch.no_grad():
        layer.weight.add_(1)
        with pytest.raises(RuntimeError, match="weights changed"):
            adapter(torch.randn((2, 64), device="cuda", dtype=torch.float16))

def test_table_fingerprint_rejected(tmp_path):
    path = policy_file(tmp_path)
    data = json.loads(path.read_text())
    data["code_fingerprint"] = "stale"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="fingerprint"):
        configure_dispatch(path)

def test_gpu_mismatch_fallback(tmp_path):
    path = policy_file(tmp_path)
    data = json.loads(path.read_text())
    data["gpu"] = "different GPU"
    path.write_text(json.dumps(data))
    configure_dispatch(path, strategy="best_custom")
    a, b = tensors()
    assert selected_implementation(a, b) == "cublaslt"

def test_invalid_strategy():
    with pytest.raises(ValueError, match="strategy"):
        configure_dispatch(None, strategy="invalid")

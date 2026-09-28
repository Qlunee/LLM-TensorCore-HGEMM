"""Integration qualification regression: failed logits must not be reported as custom success."""
from types import SimpleNamespace
import pytest
import torch
from benchmarks import bench_integration as bench


def test_failed_candidate_is_revalidated_as_explicit_fallback(monkeypatch):
    original, candidate, fallback = object(), object(), object()
    parent = SimpleNamespace(target=original)
    calls = []
    def validation(model, parent, leaf, module, prompt, tokens):
        calls.append(module)
        parent.target = module
        value = 1.004 if module is candidate else 1.0
        return [torch.tensor([value])], [{"implementation":
                "mma_async_compact" if module is candidate else "cublaslt"}]
    monkeypatch.setattr(bench, "validation_rollout", validation)
    monkeypatch.setattr(bench, "FrozenHgemmLinear", lambda layer, implementation: fallback)
    args = SimpleNamespace(max_logit_abs=0.05, max_logit_relative_l2=0.001,
                           on_logits_failure="fallback")
    module, trace, gate = bench.qualify_module(
        None, parent, "target", candidate, original, None, None,
        [torch.tensor([1.0])], "best_custom", args)
    assert calls == [candidate, fallback]
    assert module is fallback and parent.target is fallback
    assert gate["correctness"] == "pass"
    assert gate["requested_correctness"] == "fail"
    assert gate["accuracy_fallback"] == "cublaslt"
    assert gate["requested_logits_max_relative_l2_error"] > 0.001
    assert gate["executed_backends"] == [{"implementation": "cublaslt"}]


def test_strict_mode_keeps_failed_candidate(monkeypatch):
    candidate = object()
    monkeypatch.setattr(bench, "validation_rollout", lambda *args:
                        ([torch.tensor([1.004])], []))
    args = SimpleNamespace(max_logit_abs=0.05, max_logit_relative_l2=0.001,
                           on_logits_failure="error")
    module, _, gate = bench.qualify_module(None, SimpleNamespace(), "target",
        candidate, object(), None, None, [torch.tensor([1.0])], "best_custom", args)
    assert module is candidate and gate["correctness"] == "fail"
    assert gate["accuracy_fallback"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_logits_fail(value):
    result = bench.logits_gate([torch.tensor([value])], [torch.tensor([1.0])], .05, .001)
    assert result["correctness"] == "fail"
    assert result["logits_finite"] is False


def test_original_relative_gate_is_not_relaxed():
    result = bench.logits_gate([torch.tensor([1.004])], [torch.tensor([1.0])], .05, .001)
    assert result["correctness"] == "fail"
    assert result["logits_max_abs_error"] < .05
    assert result["logits_max_relative_l2_error"] > .001

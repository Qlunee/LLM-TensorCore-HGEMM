#!/usr/bin/env python3
"""Serialized model-compute timing, NOT network/service latency."""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_hgemm.dispatch import CUSTOM_PROVIDERS, configure_dispatch
from llm_hgemm.integration import FrozenHgemmLinear
from llm_hgemm.provenance import code_fingerprint, file_sha256
from benchmarks.schemas import percentile

@torch.inference_mode()
def rollout(model, prompt, tokens, collect=False):
    batch, length = prompt.shape
    trace, intervals = [], []
    mask = torch.ones_like(prompt)
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model(input_ids=prompt, attention_mask=mask,
                   use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    ttft = (time.perf_counter() - start) * 1000
    cache = output.past_key_values
    if collect:
        trace.append(output.logits.float().cpu())
    for step, token in enumerate(tokens):
        mask = torch.ones((batch, length + step + 1),
                          dtype=torch.long, device=prompt.device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model(input_ids=token, attention_mask=mask,
                       past_key_values=cache, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        intervals.append((time.perf_counter() - start) * 1000)
        cache = output.past_key_values
        if collect:
            trace.append(output.logits.float().cpu())
    return ttft, intervals, trace


def logits_gate(trace, reference, max_abs_threshold, relative_l2_threshold):
    """Keep both original gates; never treat NaN or a missing step as passing."""
    if len(trace) != len(reference) or not trace:
        raise ValueError("empty/different logits trace length")
    errors = []
    for step, (actual, expected) in enumerate(zip(trace, reference)):
        if actual.shape != expected.shape:
            raise ValueError("different logits shape")
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
        diff = actual.float() - expected.float()
        absolute = diff.abs().max().item() if finite else None
        relative = (diff.norm() / expected.float().norm().clamp_min(1e-12)).item() if finite else None
        finite = finite and math.isfinite(absolute) and math.isfinite(relative)
        if not finite:
            absolute, relative = None, None
        errors.append(dict(step=step, phase="prefill" if step == 0 else "decode",
                           finite=finite, max_abs=absolute, relative_l2=relative))
    finite = all(e["finite"] for e in errors)
    absolute = max(e["max_abs"] for e in errors) if finite else None
    relative = max(e["relative_l2"] for e in errors) if finite else None
    passed = finite and absolute <= max_abs_threshold and relative <= relative_l2_threshold
    return dict(correctness="pass" if passed else "fail", logits_finite=finite,
                logits_max_abs_error=absolute, logits_max_relative_l2_error=relative,
                logits_per_step_errors=errors)


def validation_rollout(model, parent, leaf, module, prompt, tokens):
    """Collect actual backend use only during untimed qualification."""
    setattr(parent, leaf, module)
    executed = set()
    def hook(mod, inputs, output):
        x = inputs[0]
        executed.add((x.numel() // x.shape[-1], mod.out_features,
                      mod.in_features, mod.last_backend))
    handle = (module.register_forward_hook(hook)
              if isinstance(module, FrozenHgemmLinear) else None)
    try:
        _, _, trace = rollout(model, prompt, tokens, collect=True)
    finally:
        if handle is not None:
            handle.remove()
    return trace, [dict(M=m, N=n, K=k, implementation=backend)
                   for m, n, k, backend in sorted(executed)]


def qualify_module(model, parent, leaf, module, original, prompt, tokens,
                   reference, mode, args):
    trace, executed = validation_rollout(model, parent, leaf, module, prompt, tokens)
    gate = logits_gate(trace, trace if reference is None else reference,
                       args.max_logit_abs, args.max_logit_relative_l2)
    requested = dict(gate)
    fallback = None
    if (gate["correctness"] == "fail" and mode in ("best_available", "best_custom")
            and args.on_logits_failure == "fallback"):
        fallback = "cublaslt"
        module = FrozenHgemmLinear(original, fallback)
        trace, effective_executed = validation_rollout(
            model, parent, leaf, module, prompt, tokens)
        gate = logits_gate(trace, reference, args.max_logit_abs,
                           args.max_logit_relative_l2)
        print(f"{mode}: numerical qualification FAILED; "
              f"abs={requested['logits_max_abs_error']}, "
              f"relative_l2={requested['logits_max_relative_l2_error']}; "
              "revalidating explicit cuBLASLt fallback (not custom success).")
    else:
        effective_executed = executed
    gate.update(requested_correctness=requested["correctness"],
                requested_logits_max_abs_error=requested["logits_max_abs_error"],
                requested_logits_max_relative_l2_error=requested["logits_max_relative_l2_error"],
                requested_logits_per_step_errors=requested["logits_per_step_errors"],
                requested_executed_backends=executed,
                accuracy_fallback=fallback, executed_backends=effective_executed)
    return module, trace, gate

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/lq/vllm-1p1d-v029/model/Qwen3-4B")
    p.add_argument("--target", default="model.layers.0.mlp.down_proj")
    p.add_argument("--dispatch", type=Path, required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--prompt-length", type=int, default=512)
    p.add_argument("--decode-steps", type=int, default=16)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--max-logit-abs", type=float, default=0.05)
    p.add_argument("--max-logit-relative-l2", type=float, default=0.001)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--on-logits-failure", choices=["fallback", "error"], default="fallback",
                   help="For custom strategies, revalidate/timing explicit cuBLASLt, or fail strictly. Original errors are always saved.")
    p.add_argument("--output", type=Path,
                   default=Path("results/raw/v7_integration.json"))
    args = p.parse_args()
    if min(args.batch, args.prompt_length, args.decode_steps,
           args.warmup, args.samples) <= 0:
        raise ValueError("positive workload/repetition counts required")
    if (not all(math.isfinite(v) for v in (args.max_logit_abs, args.max_logit_relative_l2))
            or min(args.max_logit_abs, args.max_logit_relative_l2) < 0):
        raise ValueError("invalid logits thresholds")
    model_path = Path(args.model).expanduser()
    is_local_model = model_path.is_dir()
    if model_path.is_absolute() and not is_local_model:
        raise FileNotFoundError(f"local model directory not found: {model_path}")
    if is_local_model:
        model_path = model_path.resolve()
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"missing config.json: {model_path}")
        index_path = model_path / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            shards = set(index.get("weight_map", {}).values())
            if not shards:
                raise ValueError(f"empty safetensors weight map: {index_path}")
            missing = [name for name in sorted(shards)
                       if not (model_path / name).is_file()
                       or (model_path / name).stat().st_size == 0]
            if missing:
                raise FileNotFoundError(f"missing/empty model shards: {missing}")
        args.model = str(model_path)
    local_files_only = args.local_files_only or is_local_model
    print(f"Model: {args.model}; local_files_only={local_files_only}; dtype=FP16")
    torch.manual_seed(2026)
    configure_dispatch(args.dispatch)
    policy_hash = file_sha256(args.dispatch)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="sdpa",
        local_files_only=local_files_only).to("cuda").eval()
    original = model.get_submodule(args.target)
    if not isinstance(original, torch.nn.Linear):
        raise ValueError("target must be nn.Linear")
    parent_name, leaf = args.target.rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    prompt = torch.randint(0, model.config.vocab_size,
                           (args.batch, args.prompt_length), device="cuda")
    tokens = [torch.randint(0, model.config.vocab_size,
                           (args.batch, 1), device="cuda")
              for _ in range(args.decode_steps)]
    modes = {
        "native": original,
        "packed_cublaslt": FrozenHgemmLinear(original, "cublaslt"),
        "best_available": FrozenHgemmLinear(original),
        "best_custom": FrozenHgemmLinear(original),
    }
    results, reference, failures = {}, None, []
    try:
        for mode, module in modes.items():
            if mode in ("best_available", "best_custom"):
                configure_dispatch(args.dispatch, strategy=mode)
            module, trace, gate = qualify_module(model, parent, leaf, module, original,
                prompt, tokens, reference, mode, args)
            if mode == "native":
                reference = trace
            if gate["correctness"] != "pass":
                results[mode] = gate
                failures.append(mode)
                print(mode, json.dumps(gate, indent=2))
                if mode == "native":
                    break
                continue  # Never time or label an unqualified path as passing.
            for _ in range(args.warmup):
                rollout(model, prompt, tokens)
            ttfts, itls, throughputs = [], [], []
            for _ in range(args.samples):
                ttft, intervals, _ = rollout(model, prompt, tokens)
                ttfts.append(ttft)
                itls.extend(intervals)
                throughputs.append(args.batch * args.decode_steps /
                                   ((ttft + sum(intervals)) / 1000))
            results[mode] = {
                "model_compute_ttft_median_ms": statistics.median(ttfts),
                "itl_median_ms": statistics.median(itls),
                "itl_p95_ms": percentile(itls, 0.95),
                "output_tokens_per_second_including_prefill":
                    statistics.median(throughputs),
                **gate,
            }
            print(mode, json.dumps(results[mode], indent=2))
    finally:
        setattr(parent, leaf, original)
    if file_sha256(args.dispatch) != policy_hash:
        raise RuntimeError("dispatch file changed during experiment")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model": args.model, "load_dtype": "float16",
        "local_files_only": local_files_only, "target": args.target,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "code_fingerprint": code_fingerprint(),
        "dispatch_sha256": policy_hash, "batch": args.batch,
        "prompt_length": args.prompt_length, "decode_steps": args.decode_steps,
        "warmup": args.warmup, "samples": args.samples,
        "on_logits_failure": args.on_logits_failure,
        "fp16_reduced_precision_reduction":
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "qualification_scope": "this model/target/fixed-token workload only; not a global dispatch guarantee",
        "custom_integration_qualified": bool(results.get("best_custom", {}).get("requested_correctness") == "pass"
            and any(e["implementation"] in CUSTOM_PROVIDERS
                    for e in results.get("best_custom", {}).get("executed_backends", []))),
        "max_logit_abs_threshold": args.max_logit_abs,
        "max_logit_relative_l2_threshold": args.max_logit_relative_l2,
        "measurement": "fixed-token serialized model-compute wall-clock",
        "results": results,
    }, indent=2, allow_nan=False))
    if failures:
        raise RuntimeError(f"Logits gate failed for {failures}; diagnostics saved to {args.output}")

if __name__ == "__main__":
    main()

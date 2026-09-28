#!/usr/bin/env python3
"""Independent validation; never counts cuBLASLt fallback as custom."""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_hgemm.dispatch import CUSTOM_PROVIDERS
from llm_hgemm.provenance import file_sha256
from benchmarks.v7_results import read_runs, times, variation

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dispatch", type=Path, required=True)
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--integration", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/raw/v7_summary.json"))
    args = p.parse_args()
    policy = json.loads(args.dispatch.read_text())
    policy_hash = file_sha256(args.dispatch)
    entries = {(e["M"], e["N"], e["K"]): e for e in policy["entries"]}
    runs, manifests = read_runs(args.inputs)
    tune_paths = {str(Path(p).resolve()) for p in policy["sources"]}
    if any(str(p.resolve()) in tune_paths for p in args.inputs):
        raise ValueError("validation must not reuse tuning CSV files")
    for manifest in manifests:
        env = manifest["environment"]
        if (env.get("dispatch_strategy") != "best_custom"
                or env.get("dispatch_sha256") != policy_hash
                or env["code_fingerprint"] != policy["code_fingerprint"]
                or (env["gpu"], env["torch"], env["torch_cuda"]) !=
                   (policy["gpu"], policy["torch"], policy["cuda"])):
            raise ValueError("validation policy/environment mismatch")
    shapes = sorted({k[:3] for k in runs[0] if k[3] == "shape_auto"})
    if not shapes or set(shapes) != set(entries):
        raise ValueError("validation must cover the entire fixed tuning set")
    details, ratios = [], []
    stable = True
    for shape in shapes:
        selected = entries[shape]["best_custom"]
        if selected not in CUSTOM_PROVIDERS:
            raise ValueError(f"not custom-covered: {shape}; remeasure, do not hide it")
        if any(run[(*shape, "shape_auto")]["selected_implementation"] != selected
               for run in runs):
            raise ValueError(f"unexpected fallback/selection: {shape}")
        custom = times(runs, (*shape, "shape_auto"))
        baseline = times(runs, (*shape, "cublaslt"))
        custom_us, baseline_us = map(statistics.median, (custom, baseline))
        ratio = baseline_us / custom_us
        ratios.append(ratio)
        shape_stable = max(variation(custom), variation(baseline)) <= policy["max_variation"]
        stable = stable and shape_stable
        details.append(dict(M=shape[0], N=shape[1], K=shape[2],
                            implementation=selected, custom_median_us=custom_us,
                            cublaslt_median_us=baseline_us, ratio_to_cublaslt=ratio,
                            custom_variation=variation(custom),
                            baseline_variation=variation(baseline), stable=shape_stable))
    integration = json.loads(args.integration.read_text())
    if (integration["dispatch_sha256"] != policy_hash
            or integration["code_fingerprint"] != policy["code_fingerprint"]
            or (integration["gpu"], integration["torch"], integration["cuda"]) !=
               (policy["gpu"], policy["torch"], policy["cuda"])):
        raise ValueError("integration policy/environment mismatch")
    modes = integration["results"]
    if any(v["correctness"] != "pass" for v in modes.values()):
        raise ValueError("integration correctness failure")
    if (modes["best_custom"].get("accuracy_fallback")
            or modes["best_custom"].get("requested_correctness", "pass") != "pass"):
        raise ValueError("custom integration failed numerical qualification; cuBLASLt fallback is NOT custom throughput")
    executed = modes["best_custom"]["executed_backends"]
    custom_executed = [e for e in executed if e["implementation"] in CUSTOM_PROVIDERS]
    if not custom_executed:
        raise ValueError("integration never executed a custom kernel")
    key = "output_tokens_per_second_including_prefill"
    native, custom = modes["native"][key], modes["best_custom"][key]
    if not all(math.isfinite(v) and v > 0 for v in (native, custom)):
        raise ValueError("invalid throughput")
    summary = {
        "shape_count": len(shapes),
        "custom_geomean_percent_of_cublaslt":
            100 * math.exp(statistics.mean(math.log(r) for r in ratios)),
        "custom_model_throughput_percent_of_native": custom / native * 100,
        "validation_stability_passed": stable,
        "resume_ready": stable,
        "integration_custom_shapes": custom_executed,
        "integration_fallback_shapes": [e for e in executed if e not in custom_executed],
        "model": integration["model"], "target": integration["target"],
        "batch": integration["batch"], "prompt_length": integration["prompt_length"],
        "decode_steps": integration["decode_steps"], "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not stable:
        raise SystemExit("Validation unstable: results saved, NOT resume-ready.")

if __name__ == "__main__":
    main()

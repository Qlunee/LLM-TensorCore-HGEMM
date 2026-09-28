#!/usr/bin/env python3
"""Offline two-policy selection; no GPU benchmark here."""
import argparse
import json
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_hgemm.dispatch import CUSTOM_PROVIDERS
from benchmarks.v7_results import read_runs, times, variation

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("configs/v7_dispatch.json"))
    p.add_argument("--min-gain", type=float, default=0.05)
    p.add_argument("--max-variation", type=float, default=0.10)
    args = p.parse_args()
    if not 0 <= args.min_gain < 1 or args.max_variation < 0:
        raise ValueError("invalid thresholds")
    runs, manifests = read_runs(args.inputs)
    if any(m["environment"].get("dispatch_sha256") for m in manifests):
        raise ValueError("tune using direct providers, not an existing policy")
    rows = []
    for shape in sorted({key[:3] for key in runs[0]}):
        reference = times(runs, (*shape, "cublaslt"))
        candidates = []
        for provider in CUSTOM_PROVIDERS:
            key = (*shape, provider)
            if key not in runs[0]:
                raise ValueError(f"missing required tuning candidate: {key}")
            values = times(runs, key)
            if variation(values) <= args.max_variation:
                candidates.append((statistics.median(values), provider, values))
        candidates.sort()
        best_custom = candidates[0][1] if candidates else "cublaslt"
        available = [item for item in candidates
                     if variation(reference) <= args.max_variation
                     and all(c <= b * (1 - args.min_gain)
                             for c, b in zip(item[2], reference))]
        best_available = available[0][1] if available else "cublaslt"
        row = dict(M=shape[0], N=shape[1], K=shape[2],
                   best_custom=best_custom, best_available=best_available,
                   custom_covered=bool(candidates),
                   custom_median_us=candidates[0][0] if candidates else None,
                   cublaslt_median_us=statistics.median(reference),
                   baseline_variation=variation(reference))
        rows.append(row)
        print(row)
    env = manifests[0]["environment"]
    payload = dict(schema_version=2, gpu=env["gpu"], sm=env["sm"],
                   cuda=env["torch_cuda"], torch=env["torch"],
                   code_fingerprint=env["code_fingerprint"],
                   min_gain=args.min_gain, max_variation=args.max_variation,
                   entries=rows, sources=[str(p.resolve()) for p in args.inputs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()

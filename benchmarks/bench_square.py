#!/usr/bin/env python3
"""Correctness-gated CUDA Event benchmark for HGEMM providers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from llm_hgemm.ops import _hgemm_out, available_providers, backend_info
from llm_hgemm.reference import correctness_passed, error_metrics, torch_reference

try:
    from benchmarks.schemas import (
        BenchmarkConfig,
        CSV_FIELDS,
        collect_environment,
        save_manifest,
        summarize_us,
    )
except ModuleNotFoundError:
    from schemas import (  # type: ignore
        BenchmarkConfig,
        CSV_FIELDS,
        collect_environment,
        save_manifest,
        summarize_us,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", nargs="+", default=["cublas", "cublaslt"])
    parser.add_argument("--shapes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--launches-per-sample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_shapes(path: Path) -> list[tuple[int, int, int]]:
    payload = json.loads(path.read_text())
    shapes = []
    for item in payload["shapes"]:
        shape = (int(item["m"]), int(item["n"]), int(item["k"]))
        if min(shape) <= 0:
            raise ValueError(f"shape dimensions must be positive: {shape}")
        shapes.append(shape)
    if not shapes:
        raise ValueError("shape configuration is empty")
    return shapes


def provider_version(provider: str) -> str:
    """Return the milestone represented by each benchmark provider."""
    versions = {
        "torch": "reference",
        "cublas": "v0-reference",
        "cublaslt": "v0-reference",
        "cutlass": "v0-reference",
        "cuda_naive": "v1.0",
        "cuda_tiled": "v1.1",
        "wmma_basic": "v2",
        "wmma_tiled": "v3",
        "mma_ptx": "v4",
        "mma_padded": "v5-padding",
        "mma_vectorized": "v5",
        "mma_double_buffer": "v6-sync",
        "mma_async": "v6",
        "mma_double_buffer_compact": "v6-compact-sync",
        "mma_async_compact": "v6-compact",
    }
    return versions[provider]


def run_provider(provider: str, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    if provider == "torch":
        torch.mm(a, b, out=out)
    else:
        _hgemm_out(a, b, out, implementation=provider)


def measure(
    provider: str,
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    config: BenchmarkConfig,
) -> list[float]:
    for _ in range(config.warmup):
        run_provider(provider, a, b, out)
    torch.cuda.synchronize()

    stream = torch.cuda.current_stream()
    samples_us: list[float] = []
    for _ in range(config.samples):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(config.launches_per_sample):
            run_provider(provider, a, b, out)
        stop.record(stream)
        stop.synchronize()
        samples_us.append(
            start.elapsed_time(stop) * 1000.0 / config.launches_per_sample
        )
    return samples_us


def make_row(
    environment: dict[str, object], provider: str, shape: tuple[int, int, int],
    config: BenchmarkConfig, stats: dict[str, float],
    metrics: dict[str, float | int], info: dict[str, int],
) -> dict[str, object]:
    m, n, k = shape
    tflops = 2.0 * m * n * k / (stats["median_us"] * 1e-6) / 1e12
    row = {field: "" for field in CSV_FIELDS}
    row.update({
        "timestamp": environment["timestamp"],
        "git_sha": environment["git_sha"],
        "gpu": environment["gpu"],
        "sm": environment["sm"],
        "driver": str(environment["nvidia_smi"]).split(",")[1].strip()
        if "," in str(environment["nvidia_smi"]) else "unknown",
        "cuda": environment["torch_cuda"],
        "torch": environment["torch"],
        "provider": provider,
        "version": provider_version(provider),
        "M": m, "N": n, "K": k,
        "layout": "row-major/row-major",
        "input_dtype": "fp16",
        "accum_dtype": "fp32",
        "output_dtype": "fp16",
        "algo_id": info.get("algorithm_id", -1),
        "workspace_bytes": info.get("workspace_bytes", 0),
        "warmup": config.warmup,
        "samples": config.samples,
        "launches_per_sample": config.launches_per_sample,
        **stats,
        "tflops": tflops,
        **metrics,
        "status": "pass" if correctness_passed(metrics) else "fail",
    })
    return row


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = BenchmarkConfig(args.warmup, args.samples, args.launches_per_sample)
    if min(config.warmup, config.samples, config.launches_per_sample) <= 0:
        raise ValueError("warmup, samples, and launches-per-sample must be positive")

    supported = set(available_providers())
    unavailable = set(args.providers) - supported
    if unavailable:
        raise ValueError(f"providers not available in this build: {sorted(unavailable)}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    environment = collect_environment()
    rows: list[dict[str, object]] = []

    for shape in load_shapes(args.shapes):
        m, n, k = shape
        a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.1
        b = torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.1
        reference = torch_reference(a, b)

        for provider in args.providers:
            out = torch.empty((m, n), device="cuda", dtype=torch.float16)
            samples_us = measure(provider, a, b, out, config)
            torch.cuda.synchronize()
            metrics = error_metrics(out, reference)
            info = backend_info(provider)
            row = make_row(
                environment, provider, shape, config,
                summarize_us(samples_us), metrics, info,
            )
            rows.append(row)
            print(
                f"{provider:8s} {m:5d}x{n:5d}x{k:5d} "
                f"median={row['median_us']:.3f} us "
                f"TFLOPS={row['tflops']:.3f} status={row['status']}"
            )

    by_shape = {(row["M"], row["N"], row["K"]): row for row in rows
                if row["provider"] == "cublaslt"}
    for row in rows:
        baseline = by_shape.get((row["M"], row["N"], row["K"]))
        if baseline:
            row["ratio_to_cublaslt"] = float(row["tflops"]) / float(baseline["tflops"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    save_manifest(args.output.with_suffix(".manifest.json"), environment, config)

    failed = [row for row in rows if row["status"] != "pass"]
    if failed:
        raise SystemExit(f"correctness gate failed for {len(failed)} result rows")


if __name__ == "__main__":
    main()

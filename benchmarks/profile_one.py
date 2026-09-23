#!/usr/bin/env python3
"""A minimal single-shape process for Nsight Compute/Systems."""

from __future__ import annotations

import argparse

import torch

from llm_hgemm.ops import _hgemm_out, available_providers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="cublaslt")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if args.provider not in available_providers() or args.provider == "torch":
        raise ValueError(f"unsupported compiled provider: {args.provider}")

    a = torch.randn((args.m, args.k), device="cuda", dtype=torch.float16) * 0.1
    b = torch.randn((args.k, args.n), device="cuda", dtype=torch.float16) * 0.1
    out = torch.empty((args.m, args.n), device="cuda", dtype=torch.float16)
    for _ in range(args.warmup):
        _hgemm_out(a, b, out, implementation=args.provider)
    torch.cuda.synchronize()
    for _ in range(args.iterations):
        _hgemm_out(a, b, out, implementation=args.provider)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

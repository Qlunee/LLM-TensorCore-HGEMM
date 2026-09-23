# LLM-TensorCore-HGEMM

Ampere Tensor Core HGEMM kernels optimized for representative LLM linear-layer shapes.

> Status: repository scaffold only. Kernel implementations and benchmark results will be added milestone by milestone (V0-V7).

## Numerical contract

- Operation: `C[M, N] = A[M, K] @ B[K, N]`
- Layout: row-major A, B, and C (NN)
- Input: FP16
- Accumulation: FP32
- Output: FP16
- Target architecture: NVIDIA Ampere SM86 (RTX 3090)

## Roadmap

- V0: correctness and benchmark harness
- V1: CUDA Core naive/tiled baselines
- V2-V3: basic and tiled WMMA
- V4-V6: MMA PTX, vectorized loads, and `cp.async` pipeline
- V7: LLM-shape dispatch and PyTorch integration

## License

GPL-3.0-only. See [LICENSE](LICENSE).

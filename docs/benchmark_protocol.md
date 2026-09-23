# V0 benchmark protocol

## Frozen numerical contract

- `C[M,N] = A[M,K] @ B[K,N]`
- A/B: FP16, row-major, contiguous
- accumulation: FP32
- C: FP16, row-major, contiguous
- `alpha=1`, `beta=0`, no TF32
- single RTX 3090 (SM86)

## Timing

- warm-up: 20 launches
- samples: 50
- launches per sample: 10
- timer: CUDA Events on the current PyTorch CUDA stream
- reported statistics: median, p95, minimum, microseconds, TFLOPS
- excluded: compilation, tensor creation, input generation, host/device copies,
  reference computation, first-time handle/heuristic creation, and CSV output

The benchmark uses preallocated output tensors. cuBLASLt plans and workspace are
cached during warm-up, so formal samples contain only repeated matmul launches.

## Correctness gate

The reference is `(A.float() @ B.float()).half()` with TF32 disabled. Every
timed result must satisfy:

- no NaN or Inf
- maximum absolute error <= `5e-2`
- relative L2 error <= `5e-3`

## Reproducibility

Each CSV row records the provider, shape, datatypes, algorithm/workspace,
sampling configuration, timings, and correctness metrics. A sibling manifest
records software versions, GPU state, Git SHA, and the frozen contract.

# LLM-TensorCore-HGEMM

English | [简体中文](README_zh.md)

Ampere Tensor Core HGEMM kernels for representative LLM linear-layer shapes.

This project develops CUDA Core, WMMA, and MMA PTX implementations on an NVIDIA RTX 3090 (SM86). It explores shared-memory tiling, vectorized copies, padding, asynchronous double buffering, and shape-aware dispatch, with a PyTorch CUDA Extension for experiments on actual Qwen3 linear layers.

The focus is **CUDA kernel engineering, controlled optimization experiments, and performance analysis**. This is not a general-purpose GEMM library, and it does not claim universal superiority over cuBLASLt or a validated full-model inference speedup.

## 1. Scope

Implemented:

- Naive and shared-memory-tiled CUDA Core baselines.
- Basic and CTA/warp-tiled WMMA kernels using `m16n16k16`.
- Warp-level MMA PTX computation using `mma.sync.aligned.m16n8k16` and `ldmatrix.sync`.
- 128-bit vectorized global-to-shared copies and shared-memory padding.
- `cp.async` double buffering with synchronous-copy ablations.
- Pipeline variants with 128×128 and 64×64 CTA tiles.
- Boundary handling and fallback paths for unsupported vector-copy alignment.
- cuBLAS, cuBLASLt, and an optional CUTLASS reference backend.
- Correctness checks, CUDA Event timing, and CSV/JSON experiment records.
- Runtime Qwen3 shape collection, offline tuning, and exact-shape dispatch.
- A frozen Linear adapter with model-level numerical qualification and explicit cuBLASLt fallback.

Not currently implemented:

- BF16, FP8, or INT8 GEMM.
- FP32 output, backward kernels, or trainable Linear replacement.
- Fused bias, SiLU, or SwiGLU epilogues.
- CuTe kernels, attention kernels, or multi-GPU GEMM.
- A validated end-to-end model speedup or complete NCU utilization results.

## 2. Numerical Contract

```text
C[M, N] = A[M, K] @ B[K, N]

A/B dtype:       FP16
Custom accum:    FP32
C dtype:         FP16
Layout:          contiguous row-major / row-major
Alpha / Beta:    1 / 0
Epilogue:        none
```

- A and B must be two-dimensional, contiguous CUDA tensors on the same device.
- M, N, and K must be positive, with matching reduction dimensions.
- Kernels execute on the current PyTorch CUDA stream.
- Custom kernels use FP32 accumulators. Different accumulation orders and intermediate rounding can produce different results; bitwise agreement is not required.
- Library results can also depend on algorithm selection and reduced-precision reduction settings. An FP32 compute-type declaration does not guarantee identical intermediate arithmetic across implementations.
- For an `nn.Linear` weight W[N, K], the adapter precomputes a contiguous Wᵀ[K, N].

Operator correctness is checked against an FP32 matmul with TF32 disabled, followed by FP16 output rounding.

## 3. Implementation Milestones

| Version | Backend | Main changes |
|---|---|---|
| V0 | `cublas`, `cublaslt`; optional `cutlass` | Numerical contract, correctness checks, and benchmarking infrastructure |
| V1 | `cuda_naive`, `cuda_tiled` | CUDA Core baselines and 16×16 shared-memory tiling |
| V2 | `wmma_basic` | Basic `m16n16k16` WMMA computation |
| V3 | `wmma_tiled` | 128×128×32 CTA tiles, 64×64 warp tiles, and boundary handling |
| V4 | `mma_ptx` | `mma.sync`, `ldmatrix`, lane-to-fragment mapping, and register-based output mapping |
| V5 | `mma_padded`, `mma_vectorized` | Shared-memory padding and 128-bit vectorized copies |
| V6 | `mma_double_buffer`, `mma_async`, and compact variants | Synchronous/asynchronous double buffering; 128×128 versus 64×64 CTA tiles |
| V7 | `shape_auto`, frozen Linear adapter | Runtime shape collection, offline tuning, dispatch, and model-level qualification |

V6 includes synchronous double-buffered ablations. Compact tiles change resource usage and data reuse, so an improvement over a different tile configuration must not be attributed entirely to `cp.async`.

## 4. Environment and Build

### Experiment environment

| Component | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 |
| Compute capability | 8.6 |
| OS | Linux |
| Python | 3.11 |
| PyTorch | 2.11.0+cu128 |
| CUDA Toolkit | 12.8 |
| Transformers | 4.51.3 |

`setup.py` explicitly generates `sm_86` code. Other GPU architectures have not served as validation platforms for this project.

### Installation

The following commands assume CUDA Toolkit 12.8 is installed at `/usr/local/cuda-12.8`. Adjust the path for your installation.

```bash
git clone https://github.com/Qlunee/LLM-TensorCore-HGEMM.git
cd LLM-TensorCore-HGEMM

conda create -n hgemm python=3.11 -y
conda activate hgemm

python -m pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  setuptools wheel ninja pytest transformers==4.51.3

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=4

nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"

python -m pip install -v --no-build-isolation -e .
```

The CUDA Version shown by `nvidia-smi` is not the version of your local `nvcc`. Check that the CUDA Toolkit used to build the extension matches the CUDA version of the PyTorch build.

Rebuild after modifying CUDA/C++ sources:

```bash
python setup.py build_ext --inplace --force
```

### Optional CUTLASS backend

Set `CUTLASS_HOME` and rebuild to enable `cutlass`:

```bash
export CUTLASS_HOME=/path/to/cutlass
python setup.py build_ext --inplace --force
```

The directory must contain `include/cutlass/cutlass.h`.

CUTLASS is an optional reference, not a dependency of the custom kernels. Its supported shapes and alignment requirements are checked through `can_implement()`. The current wrapper performs initialization on each invocation; its measured time should not be presented as the best achievable kernel-only CUTLASS performance.

## 5. Python API

```python
import torch
from llm_hgemm import hgemm

a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16) * 0.1
b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16) * 0.1

c = hgemm(a, b, implementation="mma_async_compact")
print(c.shape, c.dtype)
```

Inspect benchmark providers supported by the current build:

```python
from llm_hgemm.ops import available_providers

print(available_providers())
```

Important distinctions:

- `implementation="auto"` defaults to cuBLASLt; it does not use offline shape dispatch.
- `implementation="shape_auto"` uses a configured offline dispatch table.
- `torch` is a benchmark reference provider, not a compiled backend accepted by `hgemm()`.

Configure shape-aware dispatch:

```python
from llm_hgemm.dispatch import configure_dispatch

configure_dispatch(
    "configs/v7_dispatch_accuracy.json",
    strategy="best_available",
)

c = hgemm(a, b, implementation="shape_auto")
```

Dispatch tables are tied to the GPU, CUDA/PyTorch versions, and source/extension fingerprint. After relevant code changes or a rebuild, remeasure and regenerate the table rather than editing its fingerprint to reuse old results.

## 6. Testing and Benchmark Methodology

### Correctness tests

```bash
export CUDA_VISIBLE_DEVICES=0

python -m pytest tests -q
```

Run the V7 dispatch, adapter, and numerical-guard tests separately:

```bash
python -m pytest \
  tests/test_v7.py \
  tests/test_integration_gate.py \
  -q
```

The operator-level acceptance criteria are:

```text
NaN / Inf count:       0
Maximum absolute error <= 0.05
Relative L2 error      <= 0.005
```

These thresholds define acceptance in the current harness; they are not universal error bounds for arbitrary inputs.

### Square, rectangular, and boundary benchmarks

```bash
mkdir -p results/raw

python benchmarks/bench_square.py \
  --providers cuda_tiled wmma_tiled mma_ptx mma_vectorized \
              mma_async mma_async_compact cublaslt \
  --shapes configs/v6_shapes.json \
  --warmup 20 \
  --samples 50 \
  --launches-per-sample 1 \
  --output results/raw/square_comparison.csv
```

This configuration includes M=N=K=4096, rectangular matrices, and irregular boundaries.

Measurement protocol:

- CUDA Events on the current stream after warm-up.
- Preallocated output tensors.
- Median, p95, minimum, TFLOPS, and numerical-error reporting.
- Compilation, input generation, host/device transfers, and reference computation are excluded.
- cuBLASLt handles, plans, and workspace are initialized during warm-up and reused.
- The cuBLASLt wrapper allows up to 32 MiB of workspace and selects the first successful heuristic candidate. It does not exhaustively search every library algorithm.
- For small shapes, CUDA Event measurements can still reflect gaps between repeated submissions and should not be interpreted as theoretical compute capability.

```text
TFLOPS = 2 × M × N × K / (median_us × 1e-6) / 1e12

Performance ratio =
    median_us_cublaslt / median_us_custom
```

A ratio above 100% means the custom backend is faster. It is not a GPU-utilization metric.

### Representative recorded results

| Implementation | M=N=K | Median / μs | TFLOPS | Source |
|---|---:|---:|---:|---|
| WMMA tiled | 4096 | 6402.05 | 21.47 | `v3_formal.csv` |
| MMA PTX | 4096 | 4792.83 | 28.68 | `v4_formal.csv` |
| Vectorized MMA | 4096 | 2215.36 | 62.04 | `v5_formal.csv` |
| Compact async MMA | 2048 | 293.89 | 58.46 | Three `v6_tuned_run*.csv` runs |

The first three rows come from separate experiment batches, not a single controlled ablation. The final row uses the median of the three run-level median latencies; TFLOPS is calculated from that latency.

Across the three V6 runs:

- Compact async at M=N=K=2048 achieved **96.4%–99.7%** of the contemporaneous cuBLASLt performance.
- Using the three-run median latencies, throughput improved by **48.8%** over the single-buffer vectorized kernel.
- This improvement combines compact-tile changes with asynchronous pipelining.

Generated CSV, JSON, and profiling artifacts are ignored by `.gitignore` by default. The filenames above identify experiment outputs; they do not imply that the raw results are distributed with the repository.

## 7. V7: LLM Shapes and Dispatch

### Runtime shape collection

Forward hooks record the dimensions of Linear layers executed by Qwen3:

```text
M = number of input elements / in_features
K = in_features
N = out_features
```

Collection uses controlled random-token workloads and distinguishes Prefill from Decode with a KV cache. It is not a production request distribution or a language-quality evaluation.

Collect shapes from a local Qwen3-4B checkpoint:

```bash
export MODEL_DIR=/path/to/Qwen3-4B

python benchmarks/collect_qwen_shapes.py \
  --model "$MODEL_DIR" \
  --local-files-only \
  --batches 1 4 \
  --lengths 128 512 \
  --decode-steps 3 \
  --output configs/qwen3_shapes.csv
```

Model weights are not included in the repository.

### Current ten-shape tuning set

| M | N | K | Phase | Linear type | Included in the 83.18% subset |
|---:|---:|---:|---|---|:---:|
| 1 | 2560 | 4096 | Decode | o_proj | — |
| 1 | 2560 | 9728 | Decode | down_proj | — |
| 1 | 4096 | 2560 | Decode | q_proj | — |
| 1 | 9728 | 2560 | Decode | gate_proj | Yes |
| 4 | 4096 | 2560 | Decode | q_proj | — |
| 128 | 2560 | 4096 | Prefill | o_proj | Yes |
| 128 | 2560 | 9728 | Prefill | down_proj | — |
| 128 | 4096 | 2560 | Prefill | q_proj | Yes |
| 128 | 9728 | 2560 | Prefill | gate_proj | Yes |
| 512 | 4096 | 2560 | Prefill | q_proj | Yes |

**Performance summary:** The five marked shapes have a geometric-mean custom-to-cuBLASLt performance ratio of **83.18%**. This is a tuning-set result for that selected subset, not an aggregate over all ten shapes, an independent validation result, or a full-model throughput measurement.

### Offline tuning

```bash
for run in 1 2 3; do
  python benchmarks/bench_llm_shapes.py \
    --csv configs/qwen3_shapes.csv \
    --limit 10 \
    --providers wmma_tiled mma_ptx mma_vectorized \
                mma_async mma_async_compact cublaslt \
    --warmup 20 \
    --samples 50 \
    --launches-per-sample 10 \
    --output "results/raw/v7_tune_run${run}.csv"
done

python benchmarks/tune_dispatch.py \
  --inputs results/raw/v7_tune_run1.csv \
           results/raw/v7_tune_run2.csv \
           results/raw/v7_tune_run3.csv \
  --min-gain 0.05 \
  --max-variation 0.10 \
  --output configs/v7_dispatch_accuracy.json
```

Dispatch policies:

- `best_custom`: selects the fastest custom candidate that meets the cross-run stability requirement; falls back to cuBLASLt if none qualifies.
- `best_available`: additionally requires at least a 5% latency reduction over cuBLASLt in every run. Otherwise, it selects cuBLASLt.

Cross-run variation is defined as:

```text
variation = (max_latency - min_latency) / median_latency
```

In the latest recorded ten-shape tuning experiment, `best_available` selected cuBLASLt for every shape. Missing exact-shape entries or an unavailable valid policy normally fall back to cuBLASLt. On an eligible device, misaligned pointers use the vectorized backend's safe path.

### Model integration and numerical qualification

The frozen Linear adapter targets bias-free CUDA FP16 Linear layers:

- Transposed weights are packed ahead of inference.
- Execution is restricted to inference with gradients disabled.
- Version changes on ordinary weight tensors are detected; rebuild the adapter after modifying weights.
- Noncontiguous or empty inputs use native `F.linear`.

```bash
python benchmarks/bench_integration.py \
  --model "$MODEL_DIR" \
  --local-files-only \
  --target model.layers.0.mlp.down_proj \
  --dispatch configs/v7_dispatch_accuracy.json \
  --batch 1 \
  --prompt-length 128 \
  --decode-steps 16 \
  --warmup 3 \
  --samples 10 \
  --on-logits-failure fallback \
  --output results/raw/v7_integration.json
```

Default model-level thresholds are maximum absolute logits error ≤0.05 and maximum relative L2 error ≤0.001. Each candidate is qualified on the fixed-token sequence before timing.

Recorded limitations:

- For the current target layer, the custom candidate reached approximately 0.4032% maximum relative logits L2 error and failed the 0.1% threshold.
- The default behavior preserves the original failure, explicitly switches to cuBLASLt, and validates the fallback again.
- `requested_correctness` describes the candidate; `correctness` describes the effective execution path.
- `accuracy_fallback` and `executed_backends` identify fallback execution.
- `report_v7.py` rejects attempts to count cuBLASLt numerical-fallback throughput as a custom-backend result.

This is qualification for the benchmark's specific model, target layer, and fixed workload—not a general runtime numerical guarantee for a deployed adapter. It does not establish a full-model speedup.

### Independent validation

After generating a dispatch table, perform three additional measurements with the policy frozen:

```bash
for run in 1 2 3; do
  python benchmarks/bench_llm_shapes.py \
    --csv configs/qwen3_shapes.csv \
    --limit 10 \
    --providers shape_auto cublaslt \
    --dispatch configs/v7_dispatch_accuracy.json \
    --dispatch-strategy best_custom \
    --warmup 20 \
    --samples 50 \
    --launches-per-sample 10 \
    --output "results/raw/v7_validate_run${run}.csv"
done

python benchmarks/report_v7.py \
  --dispatch configs/v7_dispatch_accuracy.json \
  --inputs results/raw/v7_validate_run1.csv \
           results/raw/v7_validate_run2.csv \
           results/raw/v7_validate_run3.csv \
  --integration results/raw/v7_integration.json \
  --output results/raw/v7_summary.json
```

Validation must not reuse tuning CSV files. Given the currently recorded model-level numerical failure, the reporting tool will reject a qualified custom-model-throughput conclusion. This is intentional.

## 8. Nsight Compute

A single-shape profiling entry point is provided:

```bash
mkdir -p profiling/ncu

ncu \
  --target-processes all \
  --kernel-name-base function \
  --kernel-name 'regex:mma_async_kernel' \
  --launch-skip 20 \
  --launch-count 1 \
  --section LaunchStats \
  --section Occupancy \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section SchedulerStats \
  --section WarpStateStats \
  -o profiling/ncu/mma_async_compact_2048 \
  python benchmarks/profile_one.py \
    --provider mma_async_compact \
    --m 2048 --n 2048 --k 2048 \
    --warmup 20 --iterations 1
```

- GPU performance-counter access must be enabled by the administrator.
- `ERR_NVGPUCTRPERM` means utilization measurements were not obtained.
- Compare kernels under matched GPU, shape, input, and profiling conditions.
- This README does not report unmeasured Tensor Pipe, occupancy, or stall metrics.
- Tensor Pipe active-cycle percentage is not the same as achieved TFLOPS divided by theoretical peak TFLOPS.

## 9. Repository Layout

```text
LLM-TensorCore-HGEMM/
├── csrc/
│   ├── bindings.cpp
│   ├── contract.cpp
│   ├── dispatch.cpp
│   ├── common/                  # Layout, vector/async copies, CUDA utilities
│   ├── v1_cuda_naive.cu
│   ├── v1_cuda_tiled.cu
│   ├── v2_wmma_basic.cu
│   ├── v3_wmma_tiled.cu
│   ├── v4_mma_ptx.cu
│   ├── v5_mma_vectorized.cu
│   ├── v6_mma_async.cu
│   └── references/              # cuBLAS, cuBLASLt, optional CUTLASS
├── llm_hgemm/
│   ├── ops.py                   # Python HGEMM interface
│   ├── dispatch.py              # Exact-shape dispatch
│   ├── integration.py           # Frozen Linear adapter
│   ├── provenance.py            # File hashes and source fingerprints
│   └── reference.py             # Numerical reference and acceptance gates
├── benchmarks/
│   ├── bench_square.py
│   ├── bench_llm_shapes.py
│   ├── collect_qwen_shapes.py
│   ├── tune_dispatch.py
│   ├── bench_integration.py
│   ├── report_v7.py
│   ├── profile_one.py
│   ├── schemas.py
│   └── v7_results.py
├── tests/
├── configs/
├── results/
│   ├── raw/
│   └── figures/
├── profiling/
│   ├── ncu/
│   └── nsys/
├── docs/
├── setup.py
├── LICENSE
├── README.md
└── README_zh.md
```

Experiment protocols, optimization logs, mapping notes, and numerical limitations are documented in [docs](docs).

## 10. Contributing and License

Issues and pull requests are welcome, including correctness cases and optimization proposals. Performance submissions should include the GPU, software environment, shapes, numerical contract, measurement protocol, and raw results—not only a TFLOPS number.

This project is licensed under **GPL-3.0-only**. See [LICENSE](LICENSE).

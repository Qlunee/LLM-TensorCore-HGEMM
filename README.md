# LLM-TensorCore-HGEMM

English | [简体中文](README_zh.md)

FP16 Tensor Core GEMM kernels for LLM linear layers, with a PyTorch CUDA extension, exact-shape dispatch, and reproducible performance evaluation on NVIDIA Ampere GPUs.

## Performance Highlights

Measured on **NVIDIA RTX 3090 (SM86)** with FP16 inputs, FP32 accumulation, and FP16 output.

| Implementation / optimization | Recorded result | Workload |
|---|---|---|
| Vectorized MMA | **62.0 TFLOPS** | M=N=K=4096 |
| Compact asynchronous MMA | **58.5 TFLOPS**; **96.4%–99.7% of cuBLASLt throughput** across three matched runs | M=N=K=2048 |
| Compact tiling and asynchronous pipelining | **48.8% higher throughput** than the single-buffer vectorized kernel | M=N=K=2048; three-run median latencies |
| Asynchronous-copy ablation | Tensor Pipe elapsed activity **29.5% → 40.4%**, a **36.8% relative increase** | M=N=K=2048; identical 64×64 CTA tiles and double-buffer allocation |

Throughput uses CUDA Events; Tensor Pipe activity uses Nsight Compute. See [Benchmarks](#benchmarks) for measurement settings and [Profiling](#profiling) for the recorded counter data.

[Features](#features) · [Installation](#installation) · [Quick Start](#quick-start) · [Architecture](#architecture) · [Benchmarks](#benchmarks) · [Shape-Aware Dispatch](#shape-aware-dispatch) · [Model Integration](#model-integration)

## Features

- **Tensor Core compute:** WMMA and inline PTX implementations with explicit lane-to-fragment mapping, register accumulators, and `ldmatrix` shared-memory loads.
- **Memory and pipeline optimization:** 128-bit copies, padded shared-memory layouts, and `cp.async` double buffering with 128×128 and 64×64 CTA tiles. Synchronous-copy variants provide matched comparisons.
- **PyTorch integration:** a consistent GEMM interface, current-stream execution, boundary handling, and an inference-only adapter for frozen, bias-free Linear layers.
- **Offline shape selection:** policies for exact M/N/K tuples, cross-run stability checks, minimum-gain thresholds, and cuBLASLt fallback.
- **Verifiable measurements:** numerical checks, latency distributions, algorithm/workspace metadata, source/extension fingerprints, and model-level logits qualification.

The supported operation is:

```text
C[M, N] = A[M, K] @ B[K, N]
```

| Property | Contract |
|---|---|
| Inputs / accumulation / output | FP16 / FP32 custom accumulators / FP16 |
| Input layout | Two-dimensional, contiguous row-major tensors |
| Device | A and B on the same CUDA device |
| Dimensions | Positive M, N, K; matching reduction dimensions |
| Scaling / epilogue | alpha=1, beta=0, epilogue="none" |
| Execution | Current PyTorch CUDA stream; inference |

Irregular tile boundaries are supported. The current API provides forward GEMM without bias, fused activation, or backward kernels.

## Installation

### Validated environment

| Component | Version / target |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, compute capability 8.6 |
| OS / Python | Linux / Python 3.11 |
| PyTorch | 2.11.0+cu128 |
| CUDA Toolkit | 12.8 |
| Transformers, for model workflows | 4.51.3 |
| Build | C++17, CUDAExtension, Ninja |

`setup.py` currently builds `sm_86` binaries, and shape-policy loading targets SM86. Additional GPU architectures require build and dispatch changes followed by validation.

```bash
git clone https://github.com/Qlunee/LLM-TensorCore-HGEMM.git
cd LLM-TensorCore-HGEMM

conda create -n hgemm python=3.11 -y
conda activate hgemm

python -m pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install setuptools wheel ninja pytest transformers==4.51.3

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=4

nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -m pip install -v --no-build-isolation -e .
```

Adjust `CUDA_HOME` to the installed toolkit. The local `nvcc` version must match the CUDA build used by PyTorch; `nvidia-smi` reports the driver's supported CUDA version.

Rebuild the extension after changing C++ or CUDA sources:

```bash
python setup.py build_ext --inplace --force
```

To include the optional CUTLASS reference, set `CUTLASS_HOME` to a checkout containing `include/cutlass/cutlass.h` before building. Custom kernels and cuBLASLt do not depend on CUTLASS.

## Quick Start

Run a custom kernel directly:

```python
import torch
from llm_hgemm import hgemm
from llm_hgemm.reference import error_metrics, torch_reference

torch.manual_seed(2026)
a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16) * 0.1
b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16) * 0.1

with torch.inference_mode():
    c = hgemm(a, b, implementation="mma_async_compact")
    metrics = error_metrics(c, torch_reference(a, b))

print(c.shape, c.dtype)
print(metrics)
```

The reference uses FP32 matmul with TF32 disabled, followed by FP16 rounding. See [Validation](#validation) for the acceptance thresholds.

| `implementation` | Selection behavior |
|---|---|
| A backend name, such as `mma_async_compact` | Executes that implementation, including its alignment/boundary fallback |
| `auto` (default) | Uses cuBLASLt |
| `shape_auto` | Uses a configured exact-shape policy; otherwise uses cuBLASLt |

`available_providers()` lists the build's benchmark providers. The listed `torch` provider is a benchmark reference; call `torch.mm` directly rather than passing it to `hgemm`.

## Architecture

```text
PyTorch tensors / FrozenHgemmLinear
                  |
        llm_hgemm.hgemm(...)
                  |
    Explicit backend or exact-shape policy
                  |
      C++ contract and output validation
                  |
     CUDA device guard + current stream
                  |
    Custom CUDA kernel / cuBLAS / cuBLASLt
```

### Kernel implementations

All backends share the FP16 GEMM interface and benchmark harness.

| Backend | Compute and memory organization |
|---|---|
| `cuda_naive`, `cuda_tiled` | CUDA Core baselines; the tiled variant reuses 16×16 shared-memory tiles |
| `wmma_basic`, `wmma_tiled` | `m16n16k16` WMMA; the tiled variant uses 128×128×32 CTA tiles and 64×64 warp tiles |
| `mma_ptx` | `mma.sync.aligned.m16n8k16`, `ldmatrix.x4` / `ldmatrix.x2.trans`, explicit fragment mapping |
| `mma_padded`, `mma_vectorized` | Eight-element FP16 row padding; scalar-copy and 128-bit-copy variants |
| `mma_double_buffer`, `mma_async` | 128×128 CTA tiles, K step 32, synchronous or asynchronous double buffering |
| `mma_double_buffer_compact`, `mma_async_compact` | 64×64 CTA tiles and 32×32 warp tiles with the same pipeline structure |
| `cublas`, `cublaslt`, optional `cutlass` | Library reference implementations |

**Compute mapping.** A CTA stages A/B tiles in shared memory and divides its output among four warps. The PTX path maps lane registers to `m16n8k16` fragments, retaining FP32 partial sums in registers until FP16 writeback. The transposed B load forms the register layout required by MMA; global B remains row-major. See the [layout primitives](csrc/common/layout.cuh).

**Data movement.** A 128-bit copy transfers eight FP16 elements. Row padding preserves 16-byte alignment while changing shared-memory bank mapping. Aligned full tiles use specialized paths; boundary paths zero-fill input tails and mask output stores. The asynchronous path requires 16-byte-aligned pointers and N/K divisible by eight; other cases use the vectorized backend's safe path.

**Pipeline.** The asynchronous implementation alternates two shared-memory stages:

```text
Prime stage 0
  → issue next-tile cp.async copies and commit
  → compute the current tile with ldmatrix + mma.sync
  → wait_group 0 and block synchronization
  → swap stages
```

There is at most one pending copy group. Waiting after computation allows overlap while ensuring that the next tile is ready and the previous stage can be reused. Tile size balances data reuse against register/shared-memory demand. Padding and vectorization are separate mechanisms; their benefits are measured with dedicated variants. Instruction details are documented in the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) and [Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html).

## Benchmarks

### Recorded kernel performance

RTX 3090, FP16 inputs/output, FP32 custom accumulators; 20 warm-up launches, 50 samples, and one launch per sample.

| Backend | M=N=K | Median latency (µs) | TFLOPS | Local source artifact |
|---|---:|---:|---:|---|
| `wmma_tiled` | 4096 | 6402.05 | 21.47 | `v3_formal.csv` |
| `mma_ptx` | 4096 | 4792.83 | 28.68 | `v4_formal.csv` |
| `mma_vectorized` | 4096 | 2215.36 | 62.04 | `v5_formal.csv` |
| `mma_async_compact` | 2048 | 293.89 | 58.46 | Three `v6_tuned_run*.csv` runs |

The first three rows are separate measurement batches. The compact asynchronous row uses the median of three run-level medians. At 2048³, the three matched cuBLASLt ratios were **99.72%, 96.37%, and 97.21%**. Comparing three-run median latencies gives **48.8%** higher throughput than the single-buffer vectorized implementation; this combines compact tiling with asynchronous pipelining.

The named timing CSVs are local artifacts under `results/raw/` and are excluded from Git. The repository includes [NCU reports](profiling/ncu/) and the [benchmark protocol](docs/benchmark_protocol.md); the commands below generate new timing CSVs and manifests.

### Measurement protocol

The harness uses preallocated output tensors and CUDA Events on the current stream. Handle, heuristic-plan, and workspace initialization occur during warm-up. Compilation, tensor creation, host/device copies, reference computation, and result serialization are outside the timed interval. Event intervals can include submission gaps, especially for small matrices.

Each measured output is checked against FP32 matmul with TF32 disabled and FP16 rounding. Results include median, p95, minimum latency, error metrics, and backend algorithm/workspace metadata. Any numerical failure produces a nonzero benchmark exit status.

The cuBLASLt wrapper uses row-major FP16 layouts, FP32 compute type, and up to 32 MiB of workspace. It caches the first successful heuristic candidate rather than exhaustively tuning every library algorithm. CUTLASS measurements include per-call wrapper initialization.

```text
TFLOPS = 2 * M * N * K / (median_us * 1e6)
custom_to_cublaslt = median_us_cublaslt / median_us_custom
cross_run_variation = (max_latency - min_latency) / median_latency
```

A performance ratio of 100% means equal measured throughput to this cuBLASLt configuration. Different accumulation orders and library reduction settings can produce numerical differences, so qualification uses error thresholds rather than bitwise equality.

### Reproduce the kernel benchmarks

```bash
export CUDA_VISIBLE_DEVICES=0
export RUN_DIR=results/raw/local_run
mkdir -p "$RUN_DIR"

python benchmarks/bench_square.py \
  --providers cuda_tiled wmma_tiled mma_ptx mma_padded \
              mma_vectorized mma_async mma_async_compact cublaslt \
  --shapes configs/v6_shapes.json \
  --warmup 20 --samples 50 --launches-per-sample 1 \
  --output "$RUN_DIR/kernels.csv"
```

The configuration covers squares through **4096³**, rectangular shapes, alignment boundaries, and partial tiles. Each CSV has a sibling manifest containing the GPU/software environment, Git SHA, source/extension fingerprint, seed, shape-file hash, and sampling parameters.

### Qwen3-4B linear shapes

Forward hooks capture `q_proj`, `o_proj`, `gate_proj`, and `down_proj`. The [recorded shape trace](configs/qwen3_shapes.csv) supplies a ten-shape Prefill/Decode tuning set. Below, ratios use the three-run median latencies of `mma_async_compact` and cuBLASLt.

| M×N×K | Phase / layer | Custom / cuBLASLt throughput |
|---|---|---:|
| 1×2560×4096 | Decode / o_proj | 40.34% |
| 1×2560×9728 | Decode / down_proj | 34.99% |
| 1×4096×2560 | Decode / q_proj | 59.34% |
| 1×9728×2560 | Decode / gate_proj | 104.86% |
| 4×4096×2560 | Decode / q_proj | 58.28% |
| 128×2560×4096 | Prefill / o_proj | 76.96% |
| 128×2560×9728 | Prefill / down_proj | 68.04% |
| 128×4096×2560 | Prefill / q_proj | 78.95% |
| 128×9728×2560 | Prefill / gate_proj | 73.36% |
| 512×4096×2560 | Prefill / q_proj | 85.22% |

The geometric mean is **64.89% across all ten shapes**. The previously reported **83.18%** applies to the selected subset `1×9728×2560`, `128×2560×4096`, `128×4096×2560`, `128×9728×2560`, and `512×4096×2560`. These are tuning-set measurements from `v7_accuracy_tune_run{1,2,3}.csv`, not independent validation or model throughput results.

Small-M Decode exposes limited grid parallelism and partially used output tiles. Larger square-GEMM results therefore do not predict Decode efficiency. The [Decode/Prefill NCU reports](profiling/ncu/decode_prefill_20260929_163410/) document this distinction.

## Shape-Aware Dispatch

Offline tuning evaluates exact M/N/K tuples across five custom candidates and cuBLASLt. The generated table supports two policies:

| Policy | Selection rule |
|---|---|
| `best_available` (default) | Fastest stable custom candidate with at least 5% lower latency than cuBLASLt in every run; otherwise cuBLASLt |
| `best_custom` | Fastest stable custom candidate; cuBLASLt if no custom candidate qualifies |

The default cross-run variation limit is 10%. Missing shape entries use cuBLASLt; on a policy-eligible device, misaligned pointers use the vectorized backend's safe path. The recorded ten-shape table selects cuBLASLt for every `best_available` entry.

### Generate a policy

Use the checked-in shape trace to benchmark the tuning set without loading model weights:

```bash
for run in 1 2 3; do
  python benchmarks/bench_llm_shapes.py \
    --csv configs/qwen3_shapes.csv --limit 10 \
    --providers wmma_tiled mma_ptx mma_vectorized \
                mma_async mma_async_compact cublaslt \
    --warmup 20 --samples 50 --launches-per-sample 10 \
    --output "$RUN_DIR/tune_run${run}.csv" || break
done

python benchmarks/tune_dispatch.py \
  --inputs "$RUN_DIR/tune_run1.csv" \
           "$RUN_DIR/tune_run2.csv" \
           "$RUN_DIR/tune_run3.csv" \
  --min-gain 0.05 --max-variation 0.10 \
  --output "$RUN_DIR/dispatch.json"
```

The tuner requires at least three compatible runs with identical provider/shape coverage, passing correctness checks, and matching environment/provenance metadata.

### Configure and use a policy

```python
from llm_hgemm import hgemm
from llm_hgemm.dispatch import configure_dispatch

configure_dispatch("results/raw/local_run/dispatch.json",
                   strategy="best_available")
c = hgemm(a, b, implementation="shape_auto")
```

Configure the table outside the inference hot path. It binds the GPU model, SM target, CUDA/PyTorch versions, and source/extension fingerprint. Regenerate it after relevant source changes or a rebuild; the loader rejects stale fingerprints and incompatible table versions. With no eligible policy, `shape_auto` uses cuBLASLt.

## Model Integration

`FrozenHgemmLinear` adapts a bias-free CUDA FP16 `nn.Linear` by prepacking `W[N,K]` as contiguous `Wᵀ[K,N]`. It flattens input token dimensions into M, invokes HGEMM, and restores the output shape. It requires gradients disabled; noncontiguous or empty inputs use native `F.linear`. Rebuild the adapter after weight updates.

```python
import torch
from llm_hgemm.integration import FrozenHgemmLinear

linear = torch.nn.Linear(2560, 4096, bias=False,
                         device="cuda", dtype=torch.float16).eval()
adapter = FrozenHgemmLinear(linear, implementation="mma_async_compact")
x = torch.randn(1, 128, 2560, device="cuda", dtype=torch.float16) * 0.1

with torch.inference_mode():
    y = adapter(x)
print(y.shape)  # [1, 128, 4096]
```

### Collect shapes from a local checkpoint

```bash
export MODEL_DIR=/path/to/Qwen3-4B

python benchmarks/collect_qwen_shapes.py \
  --model "$MODEL_DIR" --local-files-only \
  --batches 1 4 --lengths 128 512 --decode-steps 3 \
  --output "$RUN_DIR/qwen3_shapes.csv"
```

Hooks record `M=input.numel()/in_features`, `K=in_features`, and `N=out_features` during actual Prefill and cached Decode forwards with controlled random tokens. To tune a new trace, pass this CSV to `bench_llm_shapes.py` and regenerate the policy. Model weights are supplied separately.

### Qualify a target Linear layer

```bash
python benchmarks/bench_integration.py \
  --model "$MODEL_DIR" --local-files-only \
  --target model.layers.0.mlp.down_proj \
  --dispatch "$RUN_DIR/dispatch.json" \
  --batch 1 --prompt-length 128 --decode-steps 16 \
  --warmup 3 --samples 10 \
  --on-logits-failure fallback \
  --output "$RUN_DIR/integration.json"
```

The harness compares native PyTorch, packed cuBLASLt, `best_available`, and `best_custom` execution. A fixed-token logits check precedes timing; default limits are maximum absolute error **0.05** and relative L2 error **0.001**. Failed custom candidates retain their errors and backend traces, then explicitly requalify cuBLASLt. Use `--on-logits-failure error` to reject the candidate without fallback.

For the recorded Qwen3-4B target, the custom path's maximum relative logits L2 error was **0.4032%**, above the **0.1%** limit. Its operator-level results remain separately qualified; custom full-model throughput is not qualified by this run. `requested_correctness`, `accuracy_fallback`, `executed_backends`, and `custom_integration_qualified` identify the requested and effective execution paths. See the [numerical qualification report](docs/v7_numerical_qualification.md).

## Validation

```bash
python -m pytest tests -q
```

Tests cover input/output contracts, numerical results, irregular boundaries, alignment fallback, fragment mapping, non-default streams, pipeline stage reuse, dispatch, and Linear adaptation.

| Qualification level | Acceptance criteria |
|---|---|
| GEMM output | No NaN/Inf; max absolute error ≤0.05; relative L2 error ≤0.005 against the FP32 reference rounded to FP16 |
| Target-layer model rollout | Finite logits; max absolute error ≤0.05; relative L2 error ≤0.001 against native execution |
| Offline performance selection | Passing numerical records; cross-run variation ≤10%; `best_available` also requires ≥5% latency reduction in every run |

These thresholds apply to the tested inputs and workloads. Model-level qualification is performed by the integration benchmark and is separate from adapter execution.

<details>
<summary>Independent policy validation and reporting</summary>

After tuning, collect new measurements with the policy frozen:

```bash
for run in 1 2 3; do
  python benchmarks/bench_llm_shapes.py \
    --csv configs/qwen3_shapes.csv --limit 10 \
    --providers shape_auto cublaslt \
    --dispatch "$RUN_DIR/dispatch.json" --dispatch-strategy best_custom \
    --warmup 20 --samples 50 --launches-per-sample 10 \
    --output "$RUN_DIR/validate_run${run}.csv" || break
done

python benchmarks/report_v7.py \
  --dispatch "$RUN_DIR/dispatch.json" \
  --inputs "$RUN_DIR/validate_run1.csv" \
           "$RUN_DIR/validate_run2.csv" \
           "$RUN_DIR/validate_run3.csv" \
  --integration "$RUN_DIR/integration.json" \
  --output "$RUN_DIR/summary.json"
```

The reporter requires independent validation CSVs, complete shape coverage, compatible provenance, stable measurements, and a model rollout that actually executed a numerically qualified custom backend. It rejects cuBLASLt fallback throughput as a custom result. The recorded model-level failure does not meet this reporting gate.

</details>

## Profiling

The repository includes [WMMA/PTX](profiling/ncu/ptx_20260929_152429/), [padding/vectorization](profiling/ncu/padding_vector_20260929_162220/), [synchronous/asynchronous](profiling/ncu/async_ablation_20260929_160534/), and [Decode/Prefill](profiling/ncu/decode_prefill_20260929_163410/) NCU reports.

| Matched comparison | Shape | Tensor Pipe elapsed activity | Relative increase |
|---|---|---|---:|
| WMMA tiled → MMA PTX | 4096³ | 13.14% → 17.55% | 33.5% |
| Synchronous → asynchronous compact double buffering | 2048³ | 29.52% → 40.39% | 36.84% |

Values are three-sample medians of `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`. Tensor Pipe activity describes active pipeline cycles, not achieved TFLOPS divided by theoretical peak.

Reproduce the same-tile copy comparison with GPU performance-counter access enabled:

```bash
mkdir -p "$RUN_DIR/ncu"

for provider in mma_double_buffer_compact mma_async_compact; do
  ncu \
    --target-processes all \
    --kernel-name-base function \
    --kernel-name 'regex:^mma_async_kernel' \
    --launch-skip 20 --launch-count 3 \
    --clock-control none --cache-control none \
    --section LaunchStats --section Occupancy \
    --section SchedulerStats --section WarpStateStats \
    --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed \
    -o "$RUN_DIR/ncu/${provider}_2048" \
    python benchmarks/profile_one.py \
      --provider "$provider" --m 2048 --n 2048 --k 2048 \
      --warmup 20 --iterations 3 || break
done
```

Both providers instantiate `mma_async_kernel`; template parameters distinguish synchronous and asynchronous copies. Use unprofiled CUDA Event runs for throughput reporting and NCU counters for performance attribution.

## Repository Layout

```text
csrc/             PyTorch bindings, contracts, kernel launch dispatch
  common/         Fragment layouts, vector copies, async-copy primitives
  references/     cuBLAS, cuBLASLt, optional CUTLASS wrappers
llm_hgemm/        Python API, shape policies, Linear adapter, provenance
benchmarks/       Timing, shape capture, tuning, integration, reporting
configs/          Shape traces, benchmark workloads, recorded policies
tests/            Contract, kernel, stream, dispatch, and adapter tests
profiling/ncu/    Recorded Nsight Compute reports and counter exports
results/          Locally generated timing data and figures
docs/             Benchmark protocol and numerical qualification notes
```

## License

[GPL-3.0-only](LICENSE).

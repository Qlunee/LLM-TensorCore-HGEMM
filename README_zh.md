# LLM-TensorCore-HGEMM

[English](README.md) | 简体中文

面向 LLM Linear 层典型 Shape 的 Ampere Tensor Core HGEMM 优化项目。

项目以 NVIDIA RTX 3090（SM86）为实验平台，逐步实现 CUDA Core、WMMA 和 MMA PTX 矩阵乘 Kernel，探索共享内存分块、向量化拷贝、Padding、异步双缓冲及 Shape-aware 分派，并通过 PyTorch CUDA Extension 对接真实 Qwen3 Linear 层。

本项目定位于 **CUDA Kernel 开发、优化方法验证与性能分析**，不是通用 GEMM 库，也不宣称全面超过 cuBLASLt 或已经提升完整模型的推理吞吐。

## 1. 项目范围

已实现：

- CUDA Core naive 与 shared-memory tiled 基线。
- 基于 `m16n16k16` 的基础及 Block/Warp 分块 WMMA Kernel。
- 基于 `mma.sync.aligned.m16n8k16` 和 `ldmatrix.sync` 的 Warp 级计算。
- 128-bit Global-to-Shared 向量化拷贝与共享内存 Padding。
- `cp.async` 双缓冲，以及同步拷贝消融版本。
- 128×128 与 64×64 CTA Tile 的流水线实现。
- 非规则边界处理及不满足向量化条件时的回退路径。
- cuBLAS、cuBLASLt 及可选 CUTLASS 参考后端。
- 正确性检查、CUDA Events 计时、CSV/JSON 实验记录。
- Qwen3 实际 Linear Shape 采集、离线调优与精确 Shape 分派。
- 冻结 Linear 适配，以及模型级精度检查和显式 cuBLASLt 回退。

当前不包含：

- BF16、FP8 或 INT8 GEMM。
- FP32 输出、训练反向传播或可训练 Linear 替换。
- Bias、SiLU、SwiGLU 等融合 epilogue。
- CuTe Kernel、Attention Kernel 或多 GPU GEMM。
- 已验证的端到端模型加速或完整 NCU 利用率结论。

## 2. 计算契约

```text
C[M, N] = A[M, K] @ B[K, N]

A/B dtype:       FP16
Custom accum:    FP32
C dtype:         FP16
Layout:          contiguous row-major / row-major
Alpha / Beta:    1 / 0
Epilogue:        none
```

- A、B 必须为同一 CUDA 设备上的二维连续张量。
- M、N、K 必须为正整数，且 A、B 的 K 维一致。
- Kernel 使用当前 PyTorch CUDA Stream。
- 自定义 Kernel 的累加器为 FP32；不同后端的累加顺序和中间舍入可能不同，不要求逐位一致。
- cuBLAS/PyTorch 的数值行为也可能受到算法和 reduced-precision reduction 配置影响，仅声明 FP32 compute type 不等于所有中间步骤完全相同。
- 对 `nn.Linear` 的权重 W[N, K]，适配器预先构建连续的 Wᵀ[K, N]。

项目使用 FP32 matmul、关闭 TF32，再舍入为 FP16，作为算子正确性参考。

## 3. 版本演进

| 版本 | 后端 | 主要内容 |
|---|---|---|
| V0 | `cublas`、`cublaslt`；可选 `cutlass` | 固定计算契约、正确性检查和基准测试框架 |
| V1 | `cuda_naive`、`cuda_tiled` | CUDA Core 基线；16×16 共享内存分块 |
| V2 | `wmma_basic` | 基础 `m16n16k16` WMMA 计算 |
| V3 | `wmma_tiled` | 128×128×32 CTA Tile、64×64 Warp Tile、边界处理 |
| V4 | `mma_ptx` | `mma.sync`、`ldmatrix`、Lane-to-Fragment 映射和寄存器写回 |
| V5 | `mma_padded`、`mma_vectorized` | 共享内存 Padding 与 128-bit 向量化拷贝 |
| V6 | `mma_double_buffer`、`mma_async` 及 compact 版本 | 同步/异步双缓冲；比较 128×128 与 64×64 CTA Tile |
| V7 | `shape_auto`、冻结 Linear 适配器 | 真实 Shape 采集、离线调优、分派和模型级数值验收 |

V6 同时提供同步双缓冲消融版本。紧凑 Tile 会改变资源使用和数据复用，因此不能将紧凑异步版本相对其他 Tile 的全部收益归因于 `cp.async`。

## 4. 环境与编译

### 已使用的实验环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 |
| Compute capability | 8.6 |
| 系统 | Linux |
| Python | 3.11 |
| PyTorch | 2.11.0+cu128 |
| CUDA Toolkit | 12.8 |
| Transformers | 4.51.3 |

`setup.py` 当前显式生成 `sm_86` 代码。其他 GPU 架构未作为本项目的验证平台。

### 安装

下面假设 CUDA Toolkit 12.8 已安装在 `/usr/local/cuda-12.8`；如安装位置不同，请调整路径。

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

注意：`nvidia-smi` 中的 CUDA Version 不是本机 `nvcc` 的版本。编译 Extension 时，应检查 CUDA Toolkit 与 PyTorch CUDA 构建版本是否匹配。

修改 CUDA/C++ 源码后可重新编译：

```bash
python setup.py build_ext --inplace --force
```

### 可选 CUTLASS 后端

设置 `CUTLASS_HOME` 后重新编译，才能启用 `cutlass` 后端：

```bash
export CUTLASS_HOME=/path/to/cutlass
python setup.py build_ext --inplace --force
```

该目录必须包含 `include/cutlass/cutlass.h`。

CUTLASS 是可选参考实现，不是运行自定义 Kernel 的必要依赖。其 Shape 和对齐限制由 `can_implement()` 检查；当前参考封装在每次调用时执行初始化，不应将其计时解释为仅包含底层 Kernel 的最优 CUTLASS 性能。

## 5. Python 使用方式

```python
import torch
from llm_hgemm import hgemm

a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16) * 0.1
b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16) * 0.1

c = hgemm(a, b, implementation="mma_async_compact")
print(c.shape, c.dtype)
```

查看当前构建支持的测试后端：

```python
from llm_hgemm.ops import available_providers

print(available_providers())
```

注意：

- `implementation="auto"` 默认使用 cuBLASLt，不执行离线 Shape 分派。
- `implementation="shape_auto"` 使用已配置的离线调度表。
- `torch` 是 benchmark 中的参考 provider，不是 `hgemm()` 的编译后端。

加载调度表：

```python
from llm_hgemm.dispatch import configure_dispatch

configure_dispatch(
    "configs/v7_dispatch_accuracy.json",
    strategy="best_available",
)

c = hgemm(a, b, implementation="shape_auto")
```

调度表绑定 GPU、CUDA/PyTorch 版本和代码/Extension 指纹。修改相关代码或重新编译后，应重新测量并生成调度表，不要手动替换指纹复用旧成绩。

## 6. 测试与基准协议

### 正确性测试

```bash
export CUDA_VISIBLE_DEVICES=0

python -m pytest tests -q
```

单独检查 V7 分派、适配与数值保护：

```bash
python -m pytest \
  tests/test_v7.py \
  tests/test_integration_gate.py \
  -q
```

算子验收要求：

```text
NaN / Inf count:       0
Maximum absolute error <= 0.05
Relative L2 error      <= 0.005
```

这些是当前测试框架的验收门槛，不代表任意输入都具有同样的误差上界。

### 方阵与边界基准

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

该配置包含 M=N=K=4096、矩形 Shape 和非规则边界。

测试协议：

- 在 warm-up 后使用当前 Stream 上的 CUDA Events 计时。
- 使用预分配输出张量。
- 报告 median、p95、minimum、TFLOPS 和正确性误差。
- 不计入编译、输入生成、主机与设备数据传输、参考结果计算。
- cuBLASLt 的 handle、plan 和 workspace 在 warm-up 阶段建立并复用。
- cuBLASLt 使用最多 32 MiB workspace，并选择第一个成功的 heuristic 候选，不宣称是对所有算法穷举后的最快配置。
- 小 Shape 的 CUDA Event 时间仍可能受到重复提交之间的空隙影响，不能直接解释为理论计算能力。

```text
TFLOPS = 2 × M × N × K / (median_us × 1e-6) / 1e12

Performance ratio =
    median_us_cublaslt / median_us_custom
```

性能比例大于 100% 表示自定义后端更快，不表示 GPU 利用率。

### 已记录的代表性结果

| 实现 | M=N=K | Median / μs | TFLOPS | 来源 |
|---|---:|---:|---:|---|
| WMMA tiled | 4096 | 6402.05 | 21.47 | `v3_formal.csv` |
| MMA PTX | 4096 | 4792.83 | 28.68 | `v4_formal.csv` |
| Vectorized MMA | 4096 | 2215.36 | 62.04 | `v5_formal.csv` |
| Compact async MMA | 2048 | 293.89 | 58.46 | 三轮 `v6_tuned_run*.csv` |

前三行来自不同实验批次，不应当作同一批实验的严格消融比较。最后一行使用三轮延迟中位数，TFLOPS 由该延迟计算。

在三轮 V6 实验中：

- Compact async 在 M=N=K=2048 时达到同期 cuBLASLt 性能的 **96.4%～99.7%**。
- 基于三轮延迟中位数，相对单缓冲向量化 Kernel 的吞吐提升为 **48.8%**。
- 该提升包含紧凑 Tile 调整与异步流水的联合收益。

实验生成的 CSV、JSON 和 profiling 文件默认被 `.gitignore` 忽略；上述文件名用于定位实验产物，不意味着原始结果已随仓库发布。

## 7. V7：真实 LLM Shape 与分派

### Shape 采集

通过 forward hook 读取 Qwen3 实际执行的 Linear 输入和权重维度：

```text
M = 输入元素数量 / in_features
K = in_features
N = out_features
```

采集采用受控随机 token 工作负载，区分 Prefill 与带 KV Cache 的 Decode；不是生产请求分布或自然语言质量评测。

使用本地 Qwen3-4B 模型：

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

模型权重不包含在仓库中。

### 当前十组调优 Shape

| M | N | K | 阶段 | Linear 类型 | 纳入下述 83.18% 子集 |
|---:|---:|---:|---|---|:---:|
| 1 | 2560 | 4096 | Decode | o_proj | — |
| 1 | 2560 | 9728 | Decode | down_proj | — |
| 1 | 4096 | 2560 | Decode | q_proj | — |
| 1 | 9728 | 2560 | Decode | gate_proj | 是 |
| 4 | 4096 | 2560 | Decode | q_proj | — |
| 128 | 2560 | 4096 | Prefill | o_proj | 是 |
| 128 | 2560 | 9728 | Prefill | down_proj | — |
| 128 | 4096 | 2560 | Prefill | q_proj | 是 |
| 128 | 9728 | 2560 | Prefill | gate_proj | 是 |
| 512 | 4096 | 2560 | Prefill | q_proj | 是 |

**性能摘要：** 表中标注的五组子集，其自定义后端相对 cuBLASLt 的性能比例几何平均为 **83.18%**。该值来自调优测量，不代表全部十组的整体成绩，也不是独立验证或完整模型吞吐指标。

### 离线调优

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

两种分派策略：

- `best_custom`：选择满足跨轮稳定性要求的最快自定义候选；无合格候选则回退 cuBLASLt。
- `best_available`：自定义候选除满足稳定性要求外，每轮延迟还必须比 cuBLASLt 至少降低 5%，否则使用 cuBLASLt。

跨轮波动定义为：

```text
variation = (max_latency - min_latency) / median_latency
```

当前最新十组调优记录中，`best_available` 全部选择 cuBLASLt。精确 Shape 未命中或未配置有效策略时通常回退 cuBLASLt；有效设备上的未对齐指针使用向量化后端的安全路径。

### 模型级适配与验收

冻结 Linear 适配器面向无 Bias 的 CUDA FP16 Linear：

- 预先打包转置权重。
- 仅用于无梯度推理。
- 检测普通张量的权重版本变化；修改权重后需重建适配器。
- 非连续或空输入使用原生 `F.linear`。

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

模型级默认门槛为最大 logits 绝对误差 ≤0.05、最大相对 L2 误差 ≤0.001。候选路径先完成固定 token 序列验收，再进行计时。

已记录的限制：

- 当前目标层的自定义候选最大相对 logits L2 误差约为 0.4032%，未通过 0.1% 门槛。
- 默认行为是保存原始失败数据，显式回退 cuBLASLt，并重新验收。
- `requested_correctness` 描述候选路径，`correctness` 描述最终执行路径。
- `accuracy_fallback` 和 `executed_backends` 用于识别回退。
- `report_v7.py` 不允许将 cuBLASLt 数值回退吞吐计为自定义后端成绩。

该保护属于当前 benchmark 的模型/目标层/固定工作负载验收，不是部署适配器的通用运行时数值保证。本项目不据此宣称完整模型加速。

### 独立验证

生成调度表后，另跑三轮固定策略测量：

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

验证文件不得复用调优 CSV。按当前已记录的模型级数值失败情况，报告工具会拒绝生成合格的自定义模型吞吐结论，这是预期的保护行为。

## 8. Nsight Compute

仓库提供单 Shape profiling 入口：

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

- 需要管理员授予 GPU Performance Counters 访问权限。
- 遇到 `ERR_NVGPUCTRPERM` 时，不能据此获得利用率结论。
- 比较优化前后 Kernel 时，应保持 GPU、Shape、输入和采集条件一致。
- 本 README 不报告尚未取得的 Tensor Pipe、occupancy 或 stall 指标。
- Tensor Pipe 活跃周期比例不等于实测 TFLOPS 占理论峰值的比例。

## 9. 项目结构

```text
LLM-TensorCore-HGEMM/
├── csrc/
│   ├── bindings.cpp
│   ├── contract.cpp
│   ├── dispatch.cpp
│   ├── common/                  # Layout、向量拷贝、异步拷贝和 CUDA 工具
│   ├── v1_cuda_naive.cu
│   ├── v1_cuda_tiled.cu
│   ├── v2_wmma_basic.cu
│   ├── v3_wmma_tiled.cu
│   ├── v4_mma_ptx.cu
│   ├── v5_mma_vectorized.cu
│   ├── v6_mma_async.cu
│   └── references/              # cuBLAS、cuBLASLt、可选 CUTLASS
├── llm_hgemm/
│   ├── ops.py                   # Python HGEMM 接口
│   ├── dispatch.py              # 精确 Shape 分派
│   ├── integration.py           # 冻结 Linear 适配
│   ├── provenance.py            # 文件哈希与代码指纹
│   └── reference.py             # 数值参考与验收门槛
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

实验协议、优化日志、映射说明及数值限制位于 [docs](docs)。

## 10. 贡献与许可证

欢迎通过 Issue 或 Pull Request 提交问题、正确性案例及优化建议。性能相关提交请同时说明 GPU、软件环境、Shape、数值契约、测试协议和原始结果；不要仅提供单个 TFLOPS 数字。

本项目采用 **GPL-3.0-only**，详见 [LICENSE](LICENSE)。

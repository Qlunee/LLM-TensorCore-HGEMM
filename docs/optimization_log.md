# Optimization log

本文件记录每个版本的假设、单变量改动、结果和 Nsight Compute 证据。性能数字必须来自保存的 CSV 或 NCU 报告，不预填推测值。

## 固定计算契约

- 运算：`C[M, N] = A[M, K] @ B[K, N]`
- 布局：A、B、C 均为 contiguous row-major
- 输入：FP16
- 累加：FP32
- 输出：FP16
- GPU：Ampere `sm_86`（RTX 3090）
- 正确性参考：禁用 TF32 的 PyTorch FP32 matmul，再舍入为 FP16
- 计时：warm-up 后使用当前 stream 上的 CUDA Events

## V1.0 CUDA Core naive

### 实现

- 文件：`csrc/v1_cuda_naive.cu`
- thread block：`dim3(16, 16)`，即 256 threads/block
- 每个 thread 计算一个 `C[row, col]`
- A、B 每次 K 迭代都直接从 global memory 读取
- 使用 FP32 `fmaf` 累加，最终舍入为 FP16
- 通过行列谓词支持任意正整数 M/N/K

### 假设

该版本建立最简单、可解释的自定义 CUDA 基线。它会产生大量重复 global-memory load，预计明显慢于 cuBLASLt，但能够隔离 shared-memory tiling 带来的收益。

## V1.1 CUDA Core shared-memory tiled

### 单变量改动

- 文件：`csrc/v1_cuda_tiled.cu`
- 保持 16×16 block 和“一线程一个输出”映射不变
- 每轮将 16×16 的 A/B tile 协作加载至 shared memory
- 越界元素写零，tile 前后各执行一次 `__syncthreads()`
- 每个 A/B tile 元素在 block 内最多被复用 16 次

### 假设

共享内存分块减少 A/B 的重复 global-memory 访问。中、大矩阵上，V1.1 应比 V1.0 更快；小矩阵可能因同步与搬运开销看不到收益。V1 阶段不使用 Tensor Core，因此不应声称有 Tensor Core 指令或吞吐收益。

## Benchmark results

运行 `benchmarks/bench_square.py` 后，把 `results/raw/v1_*.csv` 中的数据汇总到下表。

| M | N | K | naive median (us) | tiled median (us) | cuBLASLt median (us) | tiled / naive speedup | tiled / cuBLASLt |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 128 | TODO | TODO | TODO | TODO | TODO |
| 256 | 256 | 256 | TODO | TODO | TODO | TODO | TODO |
| 512 | 512 | 512 | TODO | TODO | TODO | TODO | TODO |
| 1024 | 1024 | 1024 | TODO | TODO | TODO | TODO | TODO |
| 2048 | 2048 | 2048 | TODO | TODO | TODO | TODO | TODO |
| 384 | 1536 | 768 | TODO | TODO | TODO | TODO | TODO |
| 127 | 129 | 63 | TODO | TODO | TODO | TODO | TODO |

## Nsight Compute evidence

固定同一 shape、输入和采样方法，分别采集 V1.0/V1.1；记录报告路径与精确 metric 名称。

| Metric | V1.0 naive | V1.1 tiled | 解释 |
|---|---:|---:|---|
| Kernel duration | TODO | TODO | 是否形成端到端 kernel 加速 |
| DRAM bytes / sectors | TODO | TODO | global-memory traffic 是否下降 |
| L1/TEX / shared throughput | TODO | TODO | shared-memory 数据复用是否生效 |
| Achieved occupancy | TODO | TODO | 资源使用是否限制并发 |
| Registers per thread | TODO | TODO | 后续优化的寄存器压力基线 |
| Warp stall reasons | TODO | TODO | 主要瓶颈是否从访存转向同步或计算 |

## 结论填写规则

1. 只引用可复现 CSV/NCU 报告中的数字。
2. 小 shape 的启动、同步开销需单独解释，不以单点结果概括全部 shape。
3. 若 tiled 未加速，先核对正确性、缓存效应、occupancy 和 stall reason，不修改结论迎合预期。
4. V1 的价值是建立 CUDA Core 基线与 shared-memory 复用证据，为 V2 WMMA Tensor Core 路径提供对照组。

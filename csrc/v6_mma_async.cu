// V6 synchronous/async two-stage HGEMM pipeline.

#include "kernels.h"

#include "common/async_copy.cuh"
#include "common/layout.cuh"
#include "common/vector_types.cuh"

#include <climits>
#include <cstdint>

#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

namespace {

using llm_hgemm::layout::kBlockK;
using llm_hgemm::layout::kThreadsPerBlock;
using llm_hgemm::layout::ldmatrix_x2_trans;
using llm_hgemm::layout::ldmatrix_x4;
using llm_hgemm::layout::mma_m16n8k16;

using llm_hgemm::vector::copy_half8;
using llm_hgemm::vector::is_aligned_16;

constexpr int kStages = 2;
constexpr int kVectorElements = 8;

// Tile dimensions are template parameters; the reference remains 128x128.
static_assert(kBlockK % kVectorElements == 0);
static_assert(kStages == 2, "Stage XOR mapping requires exactly two buffers");

// Copy one complete or zero-filled 16-byte vector.
// Invalid vectors use the valid allocation base as a dummy source.
// 统一封装“同步搬”和“异步搬”
// Async=false → V5 式 uint4 copy
// Async=true  → V6 cp.async copy
template <bool Async>
__device__ __forceinline__ void copy_vector(
    __half* destination,
    const __half* source,
    bool valid) {

    if constexpr (Async) {
        llm_hgemm::async_copy::copy_16(
            destination,
            source,
            valid ? 16 : 0);
    } else {
        if (valid) {
            copy_half8(destination, source);
        } else {
            *reinterpret_cast<uint4*>(destination) =
                make_uint4(0, 0, 0, 0);
        }
    }
}

// Each thread copies four A/B vectors for 128x128, two for 64x64.
// Only logical tile elements are written; padding is never consumed.
template <bool Async, bool BoundsChecked, int kBlockM, int kBlockN>
__device__ __forceinline__ void load_stage(
    __half* shared_a,
    __half* shared_b,
    const __half* a,
    const __half* b,
    int block_row,
    int block_column,
    int tile_k,
    int m,
    int n,
    int k,
    int thread) {

    constexpr int kAStride = kBlockK + 8;
    constexpr int kBStride = kBlockN + 8;
    // 把 A 的一个 kBlockM × kBlockK tile，切成很多个 16B 向量，然后让整个 CTA 的线程协同搬到 Shared Memory。
    // 表示 A tile 每一行能切成多少个 16B 向量，kVectorElements = 8，表示每个向量有 8 个 half 元素。
    constexpr int a_vectors_per_row = kBlockK / kVectorElements;
    // 整个 A tile 一共有多少个 16B 向量
    constexpr int a_vector_count = kBlockM * a_vectors_per_row;

#pragma unroll
    // 表示 每个线程负责若干个 vector
    for (int index = thread;
         index < a_vector_count;
         index += kThreadsPerBlock) {

        // 把一维 index 还原成 tile 内的二维位置。
        // 行来看是第几个 vector，而列号最终要变成“第几个 FP16 元素”，例：第 1 行，从第 16 个 FP16 开始，搬连续 8 个 FP16。
        const int row = index / a_vectors_per_row;
        const int column =
            (index % a_vectors_per_row) * kVectorElements;
        
        // 得到这 8 个元素在原始 Global Memory 矩阵 A 里的位置
        const int global_row = block_row + row;
        const int global_column = tile_k + column;

        bool valid = true;

        if constexpr (BoundsChecked) {
            valid =
                global_row < m &&
                global_column + kVectorElements <= k;
        }

        const __half* source = a;

        if (valid) {
            source =
                a +
                static_cast<int64_t>(global_row) * k +
                global_column;
        }

        // 把当前这个 16B 向量，从 source 搬到 Shared Memory 中 (row, column) 对应的位置。
        // kAStride 就是 Shared Memory 里 shared_a 每一行实际占多少个 FP16 元素。
        // 所有线程共享同一个 shared_a 基址->每个线程算不同的 row / column->得到不同的 destination 地址->并行搬运不同的 16B 数据块
        copy_vector<Async>(
            shared_a + row * kAStride + column,
            source,
            valid);
    }

    constexpr int b_vectors_per_row = kBlockN / kVectorElements;
    constexpr int b_vector_count = kBlockK * b_vectors_per_row;

#pragma unroll
    for (int index = thread;
         index < b_vector_count;
         index += kThreadsPerBlock) {

        const int row = index / b_vectors_per_row;
        const int column =
            (index % b_vectors_per_row) * kVectorElements;

        const int global_row = tile_k + row;
        const int global_column = block_column + column;

        bool valid = true;

        if constexpr (BoundsChecked) {
            valid =
                global_row < k &&
                global_column + kVectorElements <= n;
        }

        const __half* source = b;

        if (valid) {
            source =
                b +
                static_cast<int64_t>(global_row) * n +
                global_column;
        }

        copy_vector<Async>(
            shared_b + row * kBStride + column,
            source,
            valid);
    }

    // 把当前线程前面已经发出的若干个 cp.async 归成一个 group，并正式提交这一组异步拷贝。
    // 每线程将本 tile 的 A/B cp.async 提交为一个 group（数量由 CTA tile 决定）。
    if constexpr (Async) {
        llm_hgemm::async_copy::commit();
    }
}

// Wait for each thread's own copies, then make the completed tile available
// to all readers in the CTA.
// 先等待当前线程自己之前提交的 cp.async group 全部完成，确保它负责搬的数据已经真正写进 Shared Memory。
template <bool Async>
__device__ __forceinline__ void stage_ready() {
    if constexpr (Async) {
        // Only one committed group can be pending here. wait_group<1>()
        // would NOT guarantee this tile is ready. Waiting follows current MMA,
        // so next-tile copies have already overlapped with computation.
        llm_hgemm::async_copy::wait_group<0>();
    }

    __syncthreads();
}

// store_fragment() 就是在把每个 lane 持有的 4 个 FP32 MMA 输出，映射回 16×8 输出 tile 中对应的 4 个位置，
// 并转成 FP16 写回 Global Memory。
template <bool BoundsChecked>
__device__ __forceinline__ void store_fragment(
    const float (&accumulator)[4],
    __half* output,
    int base_row,
    int base_column,
    int m,
    int n,
    int lane) {
    
    // 一个 warp 有 32 个线程，把它分成 8 组，每组 4 个线程
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;

    // 然后每个线程有 4 个 accumulator，它们对应当前 16×8 MMA 输出 tile 中的 4 个位置
#pragma unroll
    for (int element = 0; element < 4; ++element) {
        // element 0,1 → 上半部分，第 group 行；element 2,3 → 下半部分，第 group+8 行
        const int row =
            base_row + group + (element >= 2 ? 8 : 0);

        // 也就是说每个线程负责连续两列
        const int column =
            base_column +
            thread_in_group * 2 +
            (element & 1);

        if constexpr (BoundsChecked) {
            if (row < m && column < n) {
                output[static_cast<int64_t>(row) * n + column] =
                    __float2half_rn(accumulator[element]);
            }
        } else {
            output[static_cast<int64_t>(row) * n + column] =
                __float2half_rn(accumulator[element]);
        }
    }
}

template <bool Async, bool BoundsChecked, int kBlockM, int kBlockN>
__global__ void mma_async_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int m,
    int n,
    int k) {
    constexpr int kAStride = kBlockK + 8;
    constexpr int kBStride = kBlockN + 8;
    constexpr int kWarpM = kBlockM / 2;
    constexpr int kWarpN = kBlockN / 2;
    constexpr int kMmaRows = kWarpM / 16;
    constexpr int kMmaColumns = kWarpN / 8;
    static_assert(kBlockM % 32 == 0 && kBlockN % 16 == 0);
    static_assert(kBlockN % kVectorElements == 0);
    static_assert(kAStride * sizeof(__half) % 16 == 0);
    static_assert(kBStride * sizeof(__half) % 16 == 0);
    static_assert(kStages * (kBlockM * kAStride + kBlockK * kBStride)
                      * sizeof(__half) <= 48 * 1024);
    // 准备双缓冲 Shared Memory，kStages=2 时，就是两块 ping-pong buffer
    __shared__ __align__(16)
        __half shared_a[kStages][kBlockM][kAStride];

    __shared__ __align__(16)
        __half shared_b[kStages][kBlockK][kBStride];

    const int thread = threadIdx.x;
    const int warp = thread >> 5;
    const int lane = thread & 31;

    const int block_row =
        static_cast<int>(blockIdx.y) * kBlockM;

    const int block_column =
        static_cast<int>(blockIdx.x) * kBlockN;

    const int warp_row = (warp >> 1) * kWarpM;
    const int warp_column = (warp & 1) * kWarpN;
    
    // 每 warp 有 kMmaRows × kMmaColumns 个 m16n8 子块，每 lane 每子块累加 4 个 FP32。
    float accumulator[kMmaRows][kMmaColumns][4];

#pragma unroll
    for (int mma_m = 0; mma_m < kMmaRows; ++mma_m) {
#pragma unroll
        for (int mma_n = 0; mma_n < kMmaColumns; ++mma_n) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                accumulator[mma_m][mma_n][element] = 0.0f;
            }
        }
    }

    // Prologue: Tile 0 -> Stage 0，先把第一个 K tile 从 Global Memory 搬到 Stage 0，等它完全 ready 后才能开始算
    load_stage<Async, BoundsChecked, kBlockM, kBlockN>(
        &shared_a[0][0][0],
        &shared_b[0][0][0],
        a,
        b,
        block_row,
        block_column,
        0,
        m,
        n,
        k,
        thread);

    stage_ready<Async>();

    int read_stage = 0;
    
    // 流水线主循环
    for (int tile_k = 0; tile_k < k; tile_k += kBlockK) {
        // read_stage：当前正在被 ldmatrix + mma.sync 读取、计算的那一套 buffer
        // write_stage：正在用 cp.async 搬下一块数据进去的另一套 buffer
        // read_stage = 0，write_stage = 1：用 Stage 0 计算，同时往 Stage 1 预取下一块；
        // read_stage = 1，write_stage = 0：用 Stage 1 计算，同时往 Stage 0 预取下一块；
        const int write_stage = read_stage ^ 1; //read 0 → write 1；read 1 → write 0
        //next_k 表示：下一块 A/B tile 在 K 维上的起始位置。
        const int next_k = tile_k + kBlockK;

        // This condition is uniform across the complete CTA.
        const bool has_next = next_k < k;

        if (has_next) {
            // Async copies can run while the current stage is computed.
            // The synchronous ablation uses the same buffer sequence.
            load_stage<Async, BoundsChecked, kBlockM, kBlockN>(
                //把 K 从 32 开始的下一块 tile，搬到 Stage 1
                //这里面发的是 cp.async，所以代码发完搬运请求以后，不会停在这里等数据搬完
                &shared_a[write_stage][0][0],
                &shared_b[write_stage][0][0],
                a,
                b,
                block_row,
                block_column,
                next_k,
                m,
                n,
                k,
                thread);
        }

#pragma unroll
        for (int inner_k = 0;
             inner_k < kBlockK;
             inner_k += 16) {

            uint32_t a_fragment[kMmaRows][4];

#pragma unroll
            for (int mma_m = 0; mma_m < kMmaRows; ++mma_m) {
                const int address_row =
                    warp_row + mma_m * 16 + (lane & 15);

                const int address_column =
                    inner_k + (lane >> 4) * 8;
                // 体现并行，在Stage 1 ← Tile 1 的 cp.async的同时，Stage 0已经开始运算
                ldmatrix_x4(
                    a_fragment[mma_m],
                    &shared_a[read_stage]
                             [address_row]
                             [address_column]);
            }

#pragma unroll
            for (int mma_n = 0; mma_n < kMmaColumns; ++mma_n) {
                const int address_row =
                    inner_k + (lane & 15);

                const int address_column =
                    warp_column + mma_n * 8;

                uint32_t b_fragment[2];

                ldmatrix_x2_trans(
                    b_fragment,
                    &shared_b[read_stage]
                             [address_row]
                             [address_column]);

#pragma unroll
                for (int mma_m = 0; mma_m < kMmaRows; ++mma_m) {
                    mma_m16n8k16(
                        accumulator[mma_m][mma_n],
                        a_fragment[mma_m],
                        b_fragment);
                }
            }
        }

        if (has_next) {
            // 1. Finish this thread's pending copies.
            // 2. Ensure all CTA readers finished the old stage.
            // 3. Ensure the next stage is visible to all CTA readers.
            stage_ready<Async>();

            read_stage = write_stage;
        }
    }

// 将每个 warp 累加得到的 MMA 子块写回 Global Memory
#pragma unroll
    for (int mma_m = 0; mma_m < kMmaRows; ++mma_m) {  // 遍历 M 方向的 16-row MMA tile
#pragma unroll
        for (int mma_n = 0; mma_n < kMmaColumns; ++mma_n) { // 遍历 N 方向的 8-column MMA tile
            store_fragment<BoundsChecked>(
                accumulator[mma_m][mma_n], // 当前 m16n8 tile：每线程持有 4 个 FP32 累加结果
                c, // 输出矩阵 C
                block_row + warp_row + mma_m * 16,  // 当前 MMA tile 在 C 中的起始行
                block_column + warp_column + mma_n * 8,  // 当前 MMA tile 在 C 中的起始列
                m,
                n,
                lane);  // 根据 lane 决定该线程负责写回 tile 中哪 4 个元素
        }
    }
}


//V6 的 host 端启动函数：负责把 PyTorch Tensor 转成 CUDA 指针、检查是否满足 16B 向量化条件、
// 计算 grid/block，然后根据矩阵尺寸选择“无边界检查”或“带边界检查”的 mma_async_kernel
template <bool Async, int kBlockM = 128, int kBlockN = 128>
void launch_pipeline(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream) {

    const auto* a_pointer =
        reinterpret_cast<const __half*>(
            a.data_ptr<at::Half>());

    const auto* b_pointer =
        reinterpret_cast<const __half*>(
            b.data_ptr<at::Half>());

    auto* output_pointer =
        reinterpret_cast<__half*>(
            out.data_ptr<at::Half>());

    // A partial 16-byte row vector or a misaligned pointer is handled by V5.
    const bool copy_aligned =
        a.size(1) % kVectorElements == 0 &&
        b.size(1) % kVectorElements == 0 &&
        is_aligned_16(a_pointer) &&
        is_aligned_16(b_pointer);

    if (!copy_aligned) {
        launch_mma_vectorized(a, b, out, stream);
        return;
    }

    // Keep padded coordinates and next_k safely within the int range.
    TORCH_CHECK(
        a.size(0) <= INT_MAX - kBlockM &&
        a.size(1) <= INT_MAX - kBlockK &&
        b.size(1) <= INT_MAX - kBlockN,
        "V6 dimensions exceed the supported index range");

    const int m = static_cast<int>(a.size(0));
    const int k = static_cast<int>(a.size(1));
    const int n = static_cast<int>(b.size(1));

    const dim3 block(kThreadsPerBlock);

    const dim3 grid(
        (n + kBlockN - 1) / kBlockN,
        (m + kBlockM - 1) / kBlockM);

    TORCH_CHECK(
        grid.y <= 65535,
        "V6 grid.y exceeds the CUDA launch limit");

    const bool full_tiles =
        m % kBlockM == 0 &&
        n % kBlockN == 0 &&
        k % kBlockK == 0;

    if (full_tiles) {
        mma_async_kernel<Async, false, kBlockM, kBlockN>
            <<<grid, block, 0, stream>>>(
                a_pointer,
                b_pointer,
                output_pointer,
                m,
                n,
                k);
    } else {
        mma_async_kernel<Async, true, kBlockM, kBlockN>
            <<<grid, block, 0, stream>>>(
                a_pointer,
                b_pointer,
                output_pointer,
                m,
                n,
                k);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void launch_mma_double_buffer(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream) {

    launch_pipeline<false>(a, b, out, stream);
}

void launch_mma_async(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream) {

    launch_pipeline<true>(a, b, out, stream);
}

 // Compact ablation: same padded layout, BlockK, copy/cache policy and waits.
 // Smaller tiles reduce resources but also data reuse; benchmark before dispatch.
void launch_mma_double_buffer_compact(
    const torch::Tensor& a, const torch::Tensor& b,
    torch::Tensor& out, cudaStream_t stream) {
    launch_pipeline<false, 64, 64>(a, b, out, stream);
}

void launch_mma_async_compact(
    const torch::Tensor& a, const torch::Tensor& b,
    torch::Tensor& out, cudaStream_t stream) {
    launch_pipeline<true, 64, 64>(a, b, out, stream);
}

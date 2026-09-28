// V4 direct MMA PTX and ldmatrix HGEMM kernels.

#include "kernels.h"

#include "common/layout.cuh"

#include <climits>
#include <cstdint>

#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

namespace {

using llm_hgemm::layout::kBlockK;
using llm_hgemm::layout::kBlockM;
using llm_hgemm::layout::kBlockN;
using llm_hgemm::layout::kThreadsPerBlock;
using llm_hgemm::layout::kWarpM;
using llm_hgemm::layout::kWarpN;
using llm_hgemm::layout::ldmatrix_x2_trans;
using llm_hgemm::layout::ldmatrix_x4;
using llm_hgemm::layout::mma_m16n8k16;

__device__ __forceinline__ __half load_or_zero(
    const __half* pointer,
    int row,
    int column,
    int rows,
    int columns) {

    if (row < rows && column < columns) {
        return pointer[row * columns + column];
    }

    return __float2half_rn(0.0f);
}

union PackedHalf2 {
    __half2 value;
    uint32_t bits;
};

__device__ __forceinline__ uint32_t pack_half2(
    __half first,
    __half second) {

    PackedHalf2 packed;
    packed.value = __halves2half2(first, second);
    return packed.bits;
}

// 把手写 MMA 的 4 个累加器按 lane mapping（行=`group`/`group+8`，列=`2*tig`/`2*tig+1`）转成 FP16 并写回全局内存，同时用边界判断处理非规则 shape。
__device__ __forceinline__ void store_mma_fragment(
    const float (&accumulator)[4],
    __half* output,
    int base_row,
    int base_column,
    int m,
    int n,
    int lane) {

    const int group = lane >> 2;
    const int thread_in_group = lane & 3;

    const int column = base_column + thread_in_group * 2;
    const int row_top = base_row + group;
    const int row_bottom = row_top + 8;

    if (row_top < m && column < n) {
        output[row_top * n + column] =
            __float2half_rn(accumulator[0]);
    }

    if (row_top < m && column + 1 < n) {
        output[row_top * n + column + 1] =
            __float2half_rn(accumulator[1]);
    }

    if (row_bottom < m && column < n) {
        output[row_bottom * n + column] =
            __float2half_rn(accumulator[2]);
    }

    if (row_bottom < m && column + 1 < n) {
        output[row_bottom * n + column + 1] =
            __float2half_rn(accumulator[3]);
    }
}

// -----------------------------------------------------------------------------
// V4.0: direct mma.sync probe.
//
// One warp computes one 16x8 output tile. A/B fragments are constructed
// explicitly from scalar global-memory loads. This isolates mma.sync and its
// lane-to-fragment mapping from ldmatrix.
// -----------------------------------------------------------------------------

__global__ void mma_ptx_probe_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int m,
    int n,
    int k) {

    const int lane = threadIdx.x;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;

    const int tile_row = static_cast<int>(blockIdx.y) * 16;
    const int tile_column = static_cast<int>(blockIdx.x) * 8;

    float accumulator[4] = {
        0.0f, 0.0f, 0.0f, 0.0f
    };

    for (int tile_k = 0; tile_k < k; tile_k += 16) {
        const int a_column_low =
            tile_k + thread_in_group * 2;

        const int a_column_high =
            a_column_low + 8;

        const int a_row_top =
            tile_row + group;

        const int a_row_bottom =
            a_row_top + 8;

        uint32_t a_fragment[4];

        a_fragment[0] = pack_half2(
            load_or_zero(
                a, a_row_top, a_column_low, m, k),
            load_or_zero(
                a, a_row_top, a_column_low + 1, m, k));

        a_fragment[1] = pack_half2(
            load_or_zero(
                a, a_row_bottom, a_column_low, m, k),
            load_or_zero(
                a, a_row_bottom, a_column_low + 1, m, k));

        a_fragment[2] = pack_half2(
            load_or_zero(
                a, a_row_top, a_column_high, m, k),
            load_or_zero(
                a, a_row_top, a_column_high + 1, m, k));

        a_fragment[3] = pack_half2(
            load_or_zero(
                a, a_row_bottom, a_column_high, m, k),
            load_or_zero(
                a, a_row_bottom, a_column_high + 1, m, k));

        const int b_column =
            tile_column + group;

        const int b_row_low =
            tile_k + thread_in_group * 2;

        const int b_row_high =
            b_row_low + 8;

        uint32_t b_fragment[2];

        b_fragment[0] = pack_half2(
            load_or_zero(
                b, b_row_low, b_column, k, n),
            load_or_zero(
                b, b_row_low + 1, b_column, k, n));

        b_fragment[1] = pack_half2(
            load_or_zero(
                b, b_row_high, b_column, k, n),
            load_or_zero(
                b, b_row_high + 1, b_column, k, n));

        mma_m16n8k16(
            accumulator,
            a_fragment,
            b_fragment);
    }

    store_mma_fragment(
        accumulator,
        c,
        tile_row,
        tile_column,
        m,
        n,
        lane);
}

// -----------------------------------------------------------------------------
// V4.1: ldmatrix probe.
//
// One warp computes one 16x8 tile. Global memory is staged through shared
// memory; A uses ldmatrix.x4 and B uses ldmatrix.x2.trans.
// -----------------------------------------------------------------------------
/*
先把 A/B 从 Global 协作搬到 Shared Memory，
然后用 ldmatrix.x4 加载 A 的 16×16 fragment、ldmatrix.x2.trans 加载 B 的 16×8 fragment（自动转置），再喂给 mma.sync。
相比 V4.0 的标量全局加载，ldmatrix 用一条指令替代了每 lane 多次 load，
且地址计算更简单、支持自动转置——是在验证"Shared Memory + ldmatrix + mma.sync"这条完整链路是否正确。
*/

__global__ void ldmatrix_probe_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int m,
    int n,
    int k) {

    __shared__ __align__(16) __half shared_a[16][16];
    __shared__ __align__(16) __half shared_b[16][8];

    const int lane = threadIdx.x;

    const int tile_row =
        static_cast<int>(blockIdx.y) * 16;

    const int tile_column =
        static_cast<int>(blockIdx.x) * 8;

    float accumulator[4] = {
        0.0f, 0.0f, 0.0f, 0.0f
    };

    for (int tile_k = 0; tile_k < k; tile_k += 16) {
        for (int index = lane;
             index < 16 * 16;
             index += 32) {

            const int row = index / 16;
            const int column = index % 16;

            shared_a[row][column] = load_or_zero(
                a,
                tile_row + row,
                tile_k + column,
                m,
                k);
        }

        for (int index = lane;
             index < 16 * 8;
             index += 32) {

            const int row = index / 8;
            const int column = index % 8;

            shared_b[row][column] = load_or_zero(
                b,
                tile_k + row,
                tile_column + column,
                k,
                n);
        }

        __syncwarp();

        const int a_address_row = lane & 15;
        const int a_address_column = (lane >> 4) * 8;

        uint32_t a_fragment[4];

        ldmatrix_x4(
            a_fragment,
            &shared_a[a_address_row][a_address_column]);

        // x2 needs rows 0..7 and 8..15. On sm_80+, lanes 16..31 do
        // not supply additional rows, but they still receive valid
        // duplicate addresses to keep the warp-uniform instruction safe.
        const int b_address_row = lane & 15;

        uint32_t b_fragment[2];

        ldmatrix_x2_trans(
            b_fragment,
            &shared_b[b_address_row][0]);

        mma_m16n8k16(
            accumulator,
            a_fragment,
            b_fragment);

        __syncwarp();
    }

    store_mma_fragment(
        accumulator,
        c,
        tile_row,
        tile_column,
        m,
        n,
        lane);
}

// -----------------------------------------------------------------------------
// V4.2: full CTA-tiled MMA PTX kernel.
//
// CTA tile  = 128x128x32
// Warp tile = 64x64
// MMA tile  = 16x8x16
// -----------------------------------------------------------------------------
/*
V4.2 是完整的 CTA-tiled MMA PTX kernel：
4 个 warp 协作处理 128×128×32 的 CTA tile，A/B 经 Shared Memory 中转，用 ldmatrix.x4/x2.trans 批量加载 fragment，
用 mma.sync.m16n8k16 做累加（每个 warp 64 次/K Tile），最后按 lane mapping 写回。
相比 V4.1 的单 warp 探针，V4.2 实现了多层 tile 结构、warp 协作、Shared 复用和模板化的边界处理，是一个可以实际运行的 V4 版本
*/
template <bool BoundsChecked>
__global__ void mma_ptx_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int m,
    int n,
    int k) {

    __shared__ __align__(16)
        __half shared_a[kBlockM][kBlockK];

    __shared__ __align__(16)
        __half shared_b[kBlockK][kBlockN];

    const int thread = threadIdx.x;
    const int warp = thread >> 5;
    const int lane = thread & 31;

    const int block_row =
        static_cast<int>(blockIdx.y) * kBlockM;

    const int block_column =
        static_cast<int>(blockIdx.x) * kBlockN;

    const int warp_row =
        (warp >> 1) * kWarpM;

    const int warp_column =
        (warp & 1) * kWarpN;

    float accumulator[4][8][4];

#pragma unroll
    for (int mma_m = 0; mma_m < 4; ++mma_m) {
#pragma unroll
        for (int mma_n = 0; mma_n < 8; ++mma_n) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                accumulator[mma_m][mma_n][element] = 0.0f;
            }
        }
    }

    for (int tile_k = 0; tile_k < k; tile_k += kBlockK) {
        for (int index = thread;
             index < kBlockM * kBlockK;
             index += kThreadsPerBlock) {

            const int local_row = index / kBlockK;
            const int local_column = index % kBlockK;

            const int global_row =
                block_row + local_row;

            const int global_column =
                tile_k + local_column;

            if constexpr (BoundsChecked) {
                shared_a[local_row][local_column] =
                    load_or_zero(
                        a,
                        global_row,
                        global_column,
                        m,
                        k);
            } else {
                shared_a[local_row][local_column] =
                    a[global_row * k + global_column];
            }
        }

        for (int index = thread;
             index < kBlockK * kBlockN;
             index += kThreadsPerBlock) {

            const int local_row = index / kBlockN;
            const int local_column = index % kBlockN;

            const int global_row =
                tile_k + local_row;

            const int global_column =
                block_column + local_column;

            if constexpr (BoundsChecked) {
                shared_b[local_row][local_column] =
                    load_or_zero(
                        b,
                        global_row,
                        global_column,
                        k,
                        n);
            } else {
                shared_b[local_row][local_column] =
                    b[global_row * n + global_column];
            }
        }

        __syncthreads();

#pragma unroll
        for (int inner_k = 0;
             inner_k < kBlockK;
             inner_k += 16) {

            uint32_t a_fragment[4][4];

#pragma unroll
            for (int mma_m = 0; mma_m < 4; ++mma_m) {
                const int address_row =
                    warp_row +
                    mma_m * 16 +
                    (lane & 15);

                const int address_column =
                    inner_k +
                    (lane >> 4) * 8;

                ldmatrix_x4(
                    a_fragment[mma_m],
                    &shared_a[address_row][address_column]);
            }

#pragma unroll
            for (int mma_n = 0; mma_n < 8; ++mma_n) {
                const int address_row =
                    inner_k + (lane & 15);

                const int address_column =
                    warp_column + mma_n * 8;

                uint32_t b_fragment[2];

                ldmatrix_x2_trans(
                    b_fragment,
                    &shared_b[address_row][address_column]);

#pragma unroll
                for (int mma_m = 0; mma_m < 4; ++mma_m) {
                    mma_m16n8k16(
                        accumulator[mma_m][mma_n],
                        a_fragment[mma_m],
                        b_fragment);
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int mma_m = 0; mma_m < 4; ++mma_m) {
#pragma unroll
        for (int mma_n = 0; mma_n < 8; ++mma_n) {
            const int output_row =
                block_row +
                warp_row +
                mma_m * 16;

            const int output_column =
                block_column +
                warp_column +
                mma_n * 8;

            if constexpr (BoundsChecked) {
                store_mma_fragment(
                    accumulator[mma_m][mma_n],
                    c,
                    output_row,
                    output_column,
                    m,
                    n,
                    lane);
            } else {
                const int group = lane >> 2;
                const int thread_in_group = lane & 3;

                const int column =
                    output_column +
                    thread_in_group * 2;

                const int row_top =
                    output_row + group;

                const int row_bottom =
                    row_top + 8;

                c[row_top * n + column] =
                    __float2half_rn(
                        accumulator[mma_m][mma_n][0]);

                c[row_top * n + column + 1] =
                    __float2half_rn(
                        accumulator[mma_m][mma_n][1]);

                c[row_bottom * n + column] =
                    __float2half_rn(
                        accumulator[mma_m][mma_n][2]);

                c[row_bottom * n + column + 1] =
                    __float2half_rn(
                        accumulator[mma_m][mma_n][3]);
            }
        }
    }
}

void check_dimensions(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const char* implementation) {

    TORCH_CHECK(
        a.size(0) <= INT_MAX &&
        a.size(1) <= INT_MAX &&
        b.size(1) <= INT_MAX,
        implementation,
        " supports dimensions up to INT_MAX");
}

}  // namespace

void launch_mma_ptx_probe(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream) {

    check_dimensions(a, b, "mma_ptx_probe");

    const int m = static_cast<int>(a.size(0));
    const int k = static_cast<int>(a.size(1));
    const int n = static_cast<int>(b.size(1));

    const dim3 block(32);
    const dim3 grid(
        (n + 7) / 8,
        (m + 15) / 16);

    mma_ptx_probe_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(
            a.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(
            b.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(
            out.data_ptr<at::Half>()),
        m,
        n,
        k);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_ldmatrix_probe(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream) {

    check_dimensions(a, b, "ldmatrix_probe");

    const int m = static_cast<int>(a.size(0));
    const int k = static_cast<int>(a.size(1));
    const int n = static_cast<int>(b.size(1));

    const dim3 block(32);
    const dim3 grid(
        (n + 7) / 8,
        (m + 15) / 16);

    ldmatrix_probe_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(
            a.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(
            b.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(
            out.data_ptr<at::Half>()),
        m,
        n,
        k);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_mma_ptx(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream) {

    check_dimensions(a, b, "mma_ptx");

    const int m = static_cast<int>(a.size(0));
    const int k = static_cast<int>(a.size(1));
    const int n = static_cast<int>(b.size(1));

    const dim3 block(kThreadsPerBlock);
    const dim3 grid(
        (n + kBlockN - 1) / kBlockN,
        (m + kBlockM - 1) / kBlockM);

    const bool aligned =
        m % kBlockM == 0 &&
        n % kBlockN == 0 &&
        k % kBlockK == 0;

    const auto* a_pointer =
        reinterpret_cast<const __half*>(
            a.data_ptr<at::Half>());

    const auto* b_pointer =
        reinterpret_cast<const __half*>(
            b.data_ptr<at::Half>());

    auto* output_pointer =
        reinterpret_cast<__half*>(
            out.data_ptr<at::Half>());

    if (aligned) {
        mma_ptx_kernel<false><<<grid, block, 0, stream>>>(
            a_pointer,
            b_pointer,
            output_pointer,
            m,
            n,
            k);
    } else {
        mma_ptx_kernel<true><<<grid, block, 0, stream>>>(
            a_pointer,
            b_pointer,
            output_pointer,
            m,
            n,
            k);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
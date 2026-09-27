// V3 CTA/warp tiled WMMA HGEMM.
//
// CTA tile:  128x128x32
// Warp tile:  64x64
// WMMA tile:  16x16x16
#include "kernels.h"

#include <climits>

#include <cuda_fp16.h>
#include <mma.h>

#include <c10/cuda/CUDAException.h>

namespace {

namespace wmma = nvcuda::wmma;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 32;

constexpr int kWarpM = 64;
constexpr int kWarpN = 64;

constexpr int kWarpsM = kBlockM / kWarpM;
constexpr int kWarpsN = kBlockN / kWarpN;
constexpr int kWarpsPerBlock = kWarpsM * kWarpsN;

constexpr int kWarpTilesM = kWarpM / kWmmaM;
constexpr int kWarpTilesN = kWarpN / kWmmaN;

constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock =
    kWarpsPerBlock * kWarpSize;

static_assert(kWarpsM == 2);
static_assert(kWarpsN == 2);
static_assert(kWarpsPerBlock == 4);
static_assert(kThreadsPerBlock == 128);
static_assert(kBlockK % kWmmaK == 0);

template <bool BoundsCheck>
__global__ void wmma_tiled_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int m,
    int n,
    int k) {

    __shared__ __align__(32)
        __half shared_a[kBlockM][kBlockK];

    __shared__ __align__(32)
        __half shared_b[kBlockK][kBlockN];

    // 每个 Warp 拥有独立的 FP32 输出转换区。
    __shared__ __align__(32)
        float output_scratch
            [kWarpsPerBlock]
            [kWmmaM * kWmmaN];

    const int thread_id =
        static_cast<int>(threadIdx.x);

    const int warp_id =
        thread_id / kWarpSize;

    const int lane_id =
        thread_id % kWarpSize;

    const int warp_row =
        warp_id / kWarpsN;

    const int warp_col =
        warp_id % kWarpsN;

    const int block_row =
        static_cast<int>(blockIdx.y) * kBlockM;

    const int block_col =
        static_cast<int>(blockIdx.x) * kBlockN;

    // 每个 Warp 持有一个 4×4 accumulator fragment 数组，
    // 对应一个 64×64 Warp Tile。
    wmma::fragment<
        wmma::accumulator,
        kWmmaM,
        kWmmaN,
        kWmmaK,
        float>
        accumulators[kWarpTilesM][kWarpTilesN];

#pragma unroll
    for (int warp_tile_m = 0;
         warp_tile_m < kWarpTilesM;
         ++warp_tile_m) {

#pragma unroll
        for (int warp_tile_n = 0;
             warp_tile_n < kWarpTilesN;
             ++warp_tile_n) {

            wmma::fill_fragment(
                accumulators[warp_tile_m][warp_tile_n],
                0.0f);
        }
    }

    const __half zero =
        __float2half_rn(0.0f);

    // CTA 沿 K 维以 32 为单位推进。
    for (int block_k = 0;
         block_k < k;
         block_k += kBlockK) {

        // 128 个线程协作加载 128×32 A Tile。
        for (int linear_index = thread_id;
             linear_index < kBlockM * kBlockK;
             linear_index += kThreadsPerBlock) {

            const int local_row =
                linear_index / kBlockK;

            const int local_col =
                linear_index % kBlockK;

            const int global_row =
                block_row + local_row;

            const int global_col =
                block_k + local_col;

            if constexpr (BoundsCheck) {
                if (global_row < m &&
                    global_col < k) {

                    shared_a[local_row][local_col] =
                        a[global_row * k + global_col];
                } else {
                    shared_a[local_row][local_col] =
                        zero;
                }
            } else {
                shared_a[local_row][local_col] =
                    a[global_row * k + global_col];
            }
        }

        // 128 个线程协作加载 32×128 B Tile。
        for (int linear_index = thread_id;
             linear_index < kBlockK * kBlockN;
             linear_index += kThreadsPerBlock) {

            const int local_row =
                linear_index / kBlockN;

            const int local_col =
                linear_index % kBlockN;

            const int global_row =
                block_k + local_row;

            const int global_col =
                block_col + local_col;

            if constexpr (BoundsCheck) {
                if (global_row < k &&
                    global_col < n) {

                    shared_b[local_row][local_col] =
                        b[global_row * n + global_col];
                } else {
                    shared_b[local_row][local_col] =
                        zero;
                }
            } else {
                shared_b[local_row][local_col] =
                    b[global_row * n + global_col];
            }
        }

        // Shared Memory Tile 对 CTA 内所有 Warp 可见。
        __syncthreads();

        // 一个 Block-K Tile 包含两个 WMMA K Tile：
        // [0,16) 与 [16,32)。
#pragma unroll
        for (int wmma_k = 0;
             wmma_k < kBlockK;
             wmma_k += kWmmaK) {

            wmma::fragment<
                wmma::matrix_a,
                kWmmaM,
                kWmmaN,
                kWmmaK,
                __half,
                wmma::row_major>
                a_fragments[kWarpTilesM];

            wmma::fragment<
                wmma::matrix_b,
                kWmmaM,
                kWmmaN,
                kWmmaK,
                __half,
                wmma::row_major>
                b_fragments[kWarpTilesN];

            // 当前 Warp 的4个 A fragment。
#pragma unroll
            for (int warp_tile_m = 0;
                 warp_tile_m < kWarpTilesM;
                 ++warp_tile_m) {

                const int shared_row =
                    warp_row * kWarpM +
                    warp_tile_m * kWmmaM;

                const __half* a_tile =
                    &shared_a[shared_row][wmma_k];

                wmma::load_matrix_sync(
                    a_fragments[warp_tile_m],
                    a_tile,
                    kBlockK);
            }

            // 当前 Warp 的4个 B fragment。
#pragma unroll
            for (int warp_tile_n = 0;
                 warp_tile_n < kWarpTilesN;
                 ++warp_tile_n) {

                const int shared_col =
                    warp_col * kWarpN +
                    warp_tile_n * kWmmaN;

                const __half* b_tile =
                    &shared_b[wmma_k][shared_col];

                wmma::load_matrix_sync(
                    b_fragments[warp_tile_n],
                    b_tile,
                    kBlockN);
            }

            // 4 个 A fragment × 4 个 B fragment
            // 更新 16 个 accumulator fragment。
#pragma unroll
            for (int warp_tile_m = 0;
                 warp_tile_m < kWarpTilesM;
                 ++warp_tile_m) {

#pragma unroll
                for (int warp_tile_n = 0;
                     warp_tile_n < kWarpTilesN;
                     ++warp_tile_n) {

                    wmma::mma_sync(
                        accumulators
                            [warp_tile_m]
                            [warp_tile_n],
                        a_fragments[warp_tile_m],
                        b_fragments[warp_tile_n],
                        accumulators
                            [warp_tile_m]
                            [warp_tile_n]);
                }
            }
        }

        // 所有 Warp 使用完当前 Shared Memory Tile 后，
        // 才允许下一轮覆盖 shared_a/shared_b。
        __syncthreads();
    }

    float* warp_output =
        output_scratch[warp_id];

    // 逐个 fragment 写入当前 Warp 的 FP32 scratch，
    // 再由32个 lane协作转换为 FP16 并写回。
#pragma unroll
    for (int warp_tile_m = 0;
         warp_tile_m < kWarpTilesM;
         ++warp_tile_m) {

#pragma unroll
        for (int warp_tile_n = 0;
             warp_tile_n < kWarpTilesN;
             ++warp_tile_n) {

            wmma::store_matrix_sync(
                warp_output,
                accumulators
                    [warp_tile_m]
                    [warp_tile_n],
                kWmmaN,
                wmma::mem_row_major);

            __syncwarp();

            for (int linear_index = lane_id;
                 linear_index < kWmmaM * kWmmaN;
                 linear_index += kWarpSize) {

                const int local_row =
                    linear_index / kWmmaN;

                const int local_col =
                    linear_index % kWmmaN;

                const int global_row =
                    block_row +
                    warp_row * kWarpM +
                    warp_tile_m * kWmmaM +
                    local_row;

                const int global_col =
                    block_col +
                    warp_col * kWarpN +
                    warp_tile_n * kWmmaN +
                    local_col;

                if constexpr (BoundsCheck) {
                    if (global_row < m &&
                        global_col < n) {

                        c[global_row * n + global_col] =
                            __float2half_rn(
                                warp_output[linear_index]);
                    }
                } else {
                    c[global_row * n + global_col] =
                        __float2half_rn(
                            warp_output[linear_index]);
                }
            }

            // 防止下一次 store_matrix_sync 覆盖其他 lane
            // 尚未读取完成的数据。
            __syncwarp();
        }
    }
}

}  // namespace

void launch_wmma_tiled(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream) {

    const int64_t m64 = a.size(0);
    const int64_t k64 = a.size(1);
    const int64_t n64 = b.size(1);

    TORCH_CHECK(
        m64 <= INT_MAX &&
        n64 <= INT_MAX &&
        k64 <= INT_MAX,
        "wmma_tiled supports dimensions up to INT_MAX");

    const int m = static_cast<int>(m64);
    const int n = static_cast<int>(n64);
    const int k = static_cast<int>(k64);

    const dim3 block(kThreadsPerBlock);

    const dim3 grid(
        (n + kBlockN - 1) / kBlockN,
        (m + kBlockM - 1) / kBlockM);

    const auto* a_ptr =
        reinterpret_cast<const __half*>(
            a.data_ptr<at::Half>());

    const auto* b_ptr =
        reinterpret_cast<const __half*>(
            b.data_ptr<at::Half>());

    auto* c_ptr =
        reinterpret_cast<__half*>(
            out.data_ptr<at::Half>());

    const bool fully_aligned =
        m % kBlockM == 0 &&
        n % kBlockN == 0 &&
        k % kBlockK == 0;

    if (fully_aligned) {
        wmma_tiled_kernel<false>
            <<<grid, block, 0, stream>>>(
                a_ptr,
                b_ptr,
                c_ptr,
                m,
                n,
                k);
    } else {
        wmma_tiled_kernel<true>
            <<<grid, block, 0, stream>>>(
                a_ptr,
                b_ptr,
                c_ptr,
                m,
                n,
                k);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
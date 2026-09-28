// V5 padded shared-memory layout and vectorized Global->Shared copies.

#include "kernels.h"

#include "common/layout.cuh"
#include "common/vector_types.cuh"

#include <climits>

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

using llm_hgemm::vector::copy_half8;
using llm_hgemm::vector::is_aligned_16;
using llm_hgemm::vector::kHalfElementsPerVector;

constexpr int kAPadding = 8;
constexpr int kBPadding = 8;

constexpr int kASharedStride =
    kBlockK + kAPadding;

constexpr int kBSharedStride =
    kBlockN + kBPadding;

static_assert(
    kASharedStride * sizeof(__half) % 16 == 0,
    "Every padded A row must remain 16-byte aligned");

static_assert(
    kBSharedStride * sizeof(__half) % 16 == 0,
    "Every padded B row must remain 16-byte aligned");

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

    const int column =
        base_column + thread_in_group * 2;

    const int row_top =
        base_row + group;

    const int row_bottom =
        row_top + 8;

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

template <bool VectorizedCopy, bool BoundsChecked>
__global__ void mma_vectorized_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int m,
    int n,
    int k) {

    static_assert(!VectorizedCopy || !BoundsChecked,
                  "Vector copies require complete, aligned CTA tiles");

    __shared__ __align__(16)
        __half shared_a[kBlockM][kASharedStride];

    __shared__ __align__(16)
        __half shared_b[kBlockK][kBSharedStride];

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

    // 一个线程要参与 32 个 m16n8k16 MMA 子块，而每个 MMA 子块会给该线程留下 4 个 FP32 累加结果。
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
        if constexpr (VectorizedCopy) {
            constexpr int a_vectors_per_row =
                kBlockK / kHalfElementsPerVector;

            constexpr int a_vector_count =
                kBlockM * a_vectors_per_row;

            //Block 内所有线程协同搬运
            for (int vector_index = thread;
                 vector_index < a_vector_count;
                 vector_index += kThreadsPerBlock) {

                const int local_row =
                    vector_index / a_vectors_per_row;

                const int vector_in_row =
                    vector_index % a_vectors_per_row;

                const int local_column =
                    vector_in_row * kHalfElementsPerVector;

                const int global_row =
                    block_row + local_row;

                const int global_column =
                    tile_k + local_column;
                // 每次搬 16 B
                copy_half8(
                    &shared_a[local_row][local_column],
                    &a[global_row * k + global_column]);
            }

            constexpr int b_vectors_per_row =
                kBlockN / kHalfElementsPerVector;

            constexpr int b_vector_count =
                kBlockK * b_vectors_per_row;

            for (int vector_index = thread;
                 vector_index < b_vector_count;
                 vector_index += kThreadsPerBlock) {

                const int local_row =
                    vector_index / b_vectors_per_row;

                const int vector_in_row =
                    vector_index % b_vectors_per_row;

                const int local_column =
                    vector_in_row * kHalfElementsPerVector;

                const int global_row =
                    tile_k + local_row;

                const int global_column =
                    block_column + local_column;

                copy_half8(
                    &shared_b[local_row][local_column],
                    &b[global_row * n + global_column]);
            }
        } else {
            for (int index = thread;
                 index < kBlockM * kBlockK;
                 index += kThreadsPerBlock) {

                const int local_row =
                    index / kBlockK;

                const int local_column =
                    index % kBlockK;

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

                const int local_row =
                    index / kBlockN;

                const int local_column =
                    index % kBlockN;

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
        }

        __syncthreads();

#pragma unroll
        for (int inner_k = 0;
             inner_k < kBlockK;
             inner_k += 16) {
            // 表示当前线程保存 4 组 A fragment。这里第一维 4 对应 warp 在 M 方向上的 4 个 16×... 子块，
            // 第二维 4 是每个线程为了一个 m16n8k16 的 A 操作数需要的 4 个 32-bit 寄存器。
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

// 它先用 ldmatrix 把 Shared Memory 中的 A/B tile 装进 warp 寄存器，
// 然后用大量 mma.sync.m16n8k16 完成矩阵乘加，最终把每线程持有的 4 个 FP32 accumulator 转成 FP16 写回 C。
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

bool full_tile_problem(int m, int n, int k) {
    return
        m % kBlockM == 0 &&
        n % kBlockN == 0 &&
        k % kBlockK == 0;
}

void launch_scalar_padded(
    const __half* a,
    const __half* b,
    __half* output,
    int m,
    int n,
    int k,
    dim3 grid,
    dim3 block,
    cudaStream_t stream) {

    if (full_tile_problem(m, n, k)) {
        mma_vectorized_kernel<false, false>
            <<<grid, block, 0, stream>>>(
                a, b, output, m, n, k);
    } else {
        mma_vectorized_kernel<false, true>
            <<<grid, block, 0, stream>>>(
                a, b, output, m, n, k);
    }
}

}  // namespace

void launch_mma_padded(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream) {

    check_dimensions(a, b, "mma_padded");

    const int m = static_cast<int>(a.size(0));
    const int k = static_cast<int>(a.size(1));
    const int n = static_cast<int>(b.size(1));

    const dim3 block(kThreadsPerBlock);
    const dim3 grid(
        (n + kBlockN - 1) / kBlockN,
        (m + kBlockM - 1) / kBlockM);

    launch_scalar_padded(
        reinterpret_cast<const __half*>(
            a.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(
            b.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(
            out.data_ptr<at::Half>()),
        m,
        n,
        k,
        grid,
        block,
        stream);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_mma_vectorized(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream) {

    check_dimensions(a, b, "mma_vectorized");

    const int m = static_cast<int>(a.size(0));
    const int k = static_cast<int>(a.size(1));
    const int n = static_cast<int>(b.size(1));

    const auto* a_pointer =
        reinterpret_cast<const __half*>(
            a.data_ptr<at::Half>());

    const auto* b_pointer =
        reinterpret_cast<const __half*>(
            b.data_ptr<at::Half>());

    auto* output_pointer =
        reinterpret_cast<__half*>(
            out.data_ptr<at::Half>());

    const dim3 block(kThreadsPerBlock);
    const dim3 grid(
        (n + kBlockN - 1) / kBlockN,
        (m + kBlockM - 1) / kBlockM);

    const bool vector_path =
        full_tile_problem(m, n, k) &&
        k % kHalfElementsPerVector == 0 &&
        n % kHalfElementsPerVector == 0 &&
        is_aligned_16(a_pointer) &&
        is_aligned_16(b_pointer);

    if (vector_path) {
        mma_vectorized_kernel<true, false>
            <<<grid, block, 0, stream>>>(
                a_pointer,
                b_pointer,
                output_pointer,
                m,
                n,
                k);
    } else {
        launch_scalar_padded(
            a_pointer,
            b_pointer,
            output_pointer,
            m,
            n,
            k,
            grid,
            block,
            stream);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

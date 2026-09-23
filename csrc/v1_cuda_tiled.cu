// V1 CUDA Core shared-memory tiled HGEMM baseline.
#include "kernels.h"

#include <climits>
#include <cmath>

#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr int kTile = 16;

template <int Tile>
__global__ void cuda_tiled_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int m,
    int n,
    int k) {

    __shared__ __half tile_a[Tile][Tile];
    __shared__ __half tile_b[Tile][Tile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int col =
        static_cast<int>(blockIdx.x) * Tile + tx;

    const int row =
        static_cast<int>(blockIdx.y) * Tile + ty;

    float accumulator = 0.0f;
    const __half zero = __float2half_rn(0.0f);

    for (int tile_k = 0; tile_k < k; tile_k += Tile) {
        const int a_col = tile_k + tx;
        const int b_row = tile_k + ty;

        if (row < m && a_col < k) {
            tile_a[ty][tx] =
                a[row * k + a_col];
        } else {
            // 非16对齐矩阵，越界的 A/B 元素写零
            tile_a[ty][tx] = zero;
        }

        if (b_row < k && col < n) {
            tile_b[ty][tx] =
                b[b_row * n + col];
        } else {
            tile_b[ty][tx] = zero;
        }

        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < Tile; ++inner) {
            accumulator = fmaf(
                __half2float(tile_a[ty][inner]),
                __half2float(tile_b[inner][tx]),
                accumulator);
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] =
            __float2half_rn(accumulator);
    }
}

}  // namespace

void launch_cuda_tiled(
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
        "cuda_tiled supports dimensions up to INT_MAX");

    const int m = static_cast<int>(m64);
    const int n = static_cast<int>(n64);
    const int k = static_cast<int>(k64);

    const dim3 block(kTile, kTile);

    const dim3 grid(
        (n + kTile - 1) / kTile,
        (m + kTile - 1) / kTile);

    const auto* a_ptr =
        reinterpret_cast<const __half*>(
            a.data_ptr<at::Half>());

    const auto* b_ptr =
        reinterpret_cast<const __half*>(
            b.data_ptr<at::Half>());

    auto* c_ptr =
        reinterpret_cast<__half*>(
            out.data_ptr<at::Half>());

    cuda_tiled_kernel<kTile><<<grid, block, 0, stream>>>(
        a_ptr,
        b_ptr,
        c_ptr,
        m,
        n,
        k);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
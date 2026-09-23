// V1 CUDA Core naive HGEMM baseline.
#include "kernels.h"

#include <climits>
#include <cmath>

#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

__global__ void cuda_naive_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int m,
    int n,
    int k) {

    const int col =
        static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int row =
        static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;

    if (row >= m || col >= n) {
        return;
    }

    float accumulator = 0.0f;

    for (int inner = 0; inner < k; ++inner) {
        const float a_value =
            __half2float(a[row * k + inner]);

        const float b_value =
            __half2float(b[inner * n + col]);
        // fmaf: 浮点数乘加
        accumulator = fmaf(
            a_value,
            b_value,
            accumulator);
    }

    c[row * n + col] =
        __float2half_rn(accumulator);
}

}  // namespace

void launch_cuda_naive(
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
        "cuda_naive supports dimensions up to INT_MAX");

    const int m = static_cast<int>(m64);
    const int n = static_cast<int>(n64);
    const int k = static_cast<int>(k64);

    const dim3 block(kBlockX, kBlockY);

    const dim3 grid(
        (n + block.x - 1) / block.x,
        (m + block.y - 1) / block.y);

    const auto* a_ptr =
        reinterpret_cast<const __half*>(
            a.data_ptr<at::Half>());

    const auto* b_ptr =
        reinterpret_cast<const __half*>(
            b.data_ptr<at::Half>());

    auto* c_ptr =
        reinterpret_cast<__half*>(
            out.data_ptr<at::Half>());

    cuda_naive_kernel<<<grid, block, 0, stream>>>(
        a_ptr,
        b_ptr,
        c_ptr,
        m,
        n,
        k);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
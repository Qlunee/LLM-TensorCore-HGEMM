// V2 basic WMMA m16n16k16 kernel.
// One warp computes one 16x16 output tile.
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
constexpr int kWarpSize = 32;

__global__ void wmma_basic_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int m,
    int n,
    int k) {

    // M is encoded in grid.y. Keep the explicit argument so all kernel
    // launchers expose the same M/N/K interface.
    (void)m;

    // 一个 Block 只包含一个 Warp。
    const int lane_id = static_cast<int>(threadIdx.x);

    // 当前 Warp 负责的 C Tile 左上角。
    const int tile_row =
        static_cast<int>(blockIdx.y) * kWmmaM;

    const int tile_col =
        static_cast<int>(blockIdx.x) * kWmmaN;

    /*
    段代码声明了 WMMA 运算所需的fragment：b_fragment 用来装 FP16 的 B 矩阵小块，
    accumulator_fragment 用来装 FP32 的累加结果
    它们最终会被 mma_sync 一起送入 Tensor Core，完成"FP16 输入 + FP32 累加"的矩阵乘加。
    */
    wmma::fragment<
        wmma::matrix_a,
        kWmmaM,
        kWmmaN,
        kWmmaK,
        __half,
        wmma::row_major>
        a_fragment;

    wmma::fragment<
        wmma::matrix_b,
        kWmmaM,
        kWmmaN,
        kWmmaK,
        __half,
        wmma::row_major>
        b_fragment;

    wmma::fragment<
        wmma::accumulator,
        kWmmaM,
        kWmmaN,
        kWmmaK,
        float>
        accumulator_fragment;

    wmma::fill_fragment(accumulator_fragment, 0.0f);

    // K 已由 host 保证为 16 的倍数。
    for (int tile_k = 0; tile_k < k; tile_k += kWmmaK) {
        const __half* a_tile =
            a + tile_row * k + tile_k;

        const __half* b_tile =
            b + tile_k * n + tile_col;

        // A 是 Row-major M×K，leading dimension 为 K。
        wmma::load_matrix_sync(
            a_fragment,
            a_tile,
            k);

        // B 是 Row-major K×N，leading dimension 为 N。
        wmma::load_matrix_sync(
            b_fragment,
            b_tile,
            n);

        wmma::mma_sync(
            accumulator_fragment, // D
            a_fragment,           // A
            b_fragment,           // B
            accumulator_fragment);// C
    }

    // accumulator fragment 为 FP32，先落到 FP32 Shared Memory。
    __shared__ __align__(32)
        float accumulator_tile[kWmmaM * kWmmaN];

    wmma::store_matrix_sync(
        accumulator_tile,
        accumulator_fragment,
        kWmmaN,
        wmma::mem_row_major);

    // store_matrix_sync 是 Warp Collective。
    // 在读取 Shared Memory 前显式同步完整 Warp。
    /*
    store_matrix_sync 把寄存器里的 fragment 协作写回共享内存，
    __syncwarp() 则确保 warp 内 32 个线程的写入全部完成且对彼此可见
    因为接下来可能有线程要读别人写的数据。这是 WMMA warp 级协作后的标准同步动作。
    */
    __syncwarp();

    // 256 个结果由 32 个 lane 分摊，每个 lane 转换 8 个元素。
    for (int linear_index = lane_id;
         linear_index < kWmmaM * kWmmaN;
         linear_index += kWarpSize) {

        const int local_row =
            linear_index / kWmmaN;

        const int local_col =
            linear_index % kWmmaN;

        const int global_row =
            tile_row + local_row;

        const int global_col =
            tile_col + local_col;

        c[global_row * n + global_col] =
            __float2half_rn(
                accumulator_tile[linear_index]);
    }
}

}  // namespace

void launch_wmma_basic(
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
        "wmma_basic supports dimensions up to INT_MAX");

    // V2 只让完全对齐的 Shape 进入 WMMA。
    // 非对齐 Shape 使用已经验证正确的 V1 Tiled fallback。
    const bool wmma_aligned =
        m64 % kWmmaM == 0 &&
        n64 % kWmmaN == 0 &&
        k64 % kWmmaK == 0;

    if (!wmma_aligned) {
        launch_cuda_tiled(
            a,
            b,
            out,
            stream);
        return;
    }

    const int m = static_cast<int>(m64);
    const int n = static_cast<int>(n64);
    const int k = static_cast<int>(k64);

    const dim3 block(kWarpSize);

    const dim3 grid(
        n / kWmmaN,
        m / kWmmaM);

    const auto* a_ptr =
        reinterpret_cast<const __half*>(
            a.data_ptr<at::Half>());

    const auto* b_ptr =
        reinterpret_cast<const __half*>(
            b.data_ptr<at::Half>());

    auto* c_ptr =
        reinterpret_cast<__half*>(
            out.data_ptr<at::Half>());

    wmma_basic_kernel<<<grid, block, 0, stream>>>(
        a_ptr,
        b_ptr,
        c_ptr,
        m,
        n,
        k);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

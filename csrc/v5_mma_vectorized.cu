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
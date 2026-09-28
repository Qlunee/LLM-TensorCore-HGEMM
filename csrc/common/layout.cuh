/*
V4 的底层原语：定义分块常量，
并用内联 PTX 封装三个操作——ldmatrix_x4 加载 A 分片、ldmatrix_x2_trans 加载 B 分片、mma_m16n8k16 执行一次 Tensor Core 矩阵乘加。
*/
#pragma once

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llm_hgemm::layout {

constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 16;

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 32;

constexpr int kWarpM = 64;
constexpr int kWarpN = 64;

constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;

__device__ __forceinline__ uint32_t shared_address(
    const void* pointer) {

    return static_cast<uint32_t>(
        __cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t (&destination)[4],
    const void* pointer) {

    const uint32_t address = shared_address(pointer);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(destination[0]),
          "=r"(destination[1]),
          "=r"(destination[2]),
          "=r"(destination[3])
        : "r"(address));
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t (&destination)[2],
    const void* pointer) {

    const uint32_t address = shared_address(pointer);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(destination[0]),
          "=r"(destination[1])
        : "r"(address));
}

__device__ __forceinline__ void mma_m16n8k16(
    float (&accumulator)[4],
    const uint32_t (&a_fragment)[4],
    const uint32_t (&b_fragment)[2]) {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(accumulator[0]),
          "+f"(accumulator[1]),
          "+f"(accumulator[2]),
          "+f"(accumulator[3])
        : "r"(a_fragment[0]),
          "r"(a_fragment[1]),
          "r"(a_fragment[2]),
          "r"(a_fragment[3]),
          "r"(b_fragment[0]),
          "r"(b_fragment[1]));
#endif
}

}  // namespace llm_hgemm::layout

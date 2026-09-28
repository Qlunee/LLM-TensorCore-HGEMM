/*
提供一套 16B 对齐的 FP16 向量化拷贝工具，包括检查地址是否 16B 对齐，并用 uint4 一次搬运 8 个 FP16 数据。
*/
#pragma once

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llm_hgemm::vector {

constexpr int kVectorBytes = 16;
constexpr int kHalfElementsPerVector = 8;

static_assert(
    sizeof(uint4) == kVectorBytes,
    "uint4 must represent one 128-bit vector");

static_assert(
    alignof(uint4) == kVectorBytes,
    "uint4 must have 16-byte alignment");

inline bool is_aligned_16(const void* pointer) noexcept {
    return (
        reinterpret_cast<std::uintptr_t>(pointer) &
        (kVectorBytes - 1)
    ) == 0;
}

__device__ __forceinline__ void copy_half8(
    __half* destination,
    const __half* source) {

    const uint4 value =
        *reinterpret_cast<const uint4*>(source);

    *reinterpret_cast<uint4*>(destination) = value;
}

}  // namespace llm_hgemm::vector
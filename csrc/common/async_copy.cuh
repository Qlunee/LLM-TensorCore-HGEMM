/*
这个程序封装了 cp.async 的 16B Global→Shared 异步拷贝，以及对应的 commit 和 wait_all 操作，用于构建访存与计算重叠的流水线。
*/
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace llm_hgemm::async_copy {

// source_bytes must be 0 or 16 in this project's first async implementation.
// A zero-byte source produces a complete 16-byte zero-filled destination.
__device__ __forceinline__ void copy_16(
    void* destination,
    const void* source,
    int source_bytes) {

    // 把普通 C++ 指针 destination 转成 Shared Memory 地址空间里的地址
    const uint32_t shared_pointer =
        static_cast<uint32_t>(
            __cvta_generic_to_shared(destination));

    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
        :
        : "r"(shared_pointer),
          "l"(source),
          "r"(source_bytes)
        : "memory");
}

__device__ __forceinline__ void commit() {
    asm volatile(
        "cp.async.commit_group;\n"
        :
        :
        : "memory");
}

template <int Pending>
__device__ __forceinline__ void wait_group() {
    static_assert(Pending >= 0 && Pending <= 7);
    asm volatile(
        "cp.async.wait_group %0;\n"
        :
        : "n"(Pending)
        : "memory");
}

// Compatibility alias: waits for committed groups, does not commit copies.
__device__ __forceinline__ void wait_all() {
    wait_group<0>();
}

}  // namespace llm_hgemm::async_copy

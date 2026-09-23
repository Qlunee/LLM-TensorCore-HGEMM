#include "reference.h"

#ifdef LLM_HGEMM_WITH_CUTLASS
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#endif

namespace {
thread_local ReferenceBackendInfo last_info{-1, 0};
}

bool cutlass_reference_available() {
#ifdef LLM_HGEMM_WITH_CUTLASS
    return true;
#else
    return false;
#endif
}

void launch_cutlass_reference(const torch::Tensor& a, const torch::Tensor& b,
                              torch::Tensor& out, cudaStream_t stream) {
#ifdef LLM_HGEMM_WITH_CUTLASS
    using Element = cutlass::half_t;
    using Layout = cutlass::layout::RowMajor;
    using Accumulator = float;
    using Gemm = cutlass::gemm::device::Gemm<
        Element, Layout, Element, Layout, Element, Layout, Accumulator,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            Element, 8, Accumulator, Accumulator>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2, 8, 8>;

    const int64_t m = a.size(0);
    const int64_t k = a.size(1);
    const int64_t n = b.size(1);
    typename Gemm::Arguments arguments(
        {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)},
        {reinterpret_cast<const Element*>(a.data_ptr<at::Half>()), static_cast<int>(k)},
        {reinterpret_cast<const Element*>(b.data_ptr<at::Half>()), static_cast<int>(n)},
        {reinterpret_cast<const Element*>(out.data_ptr<at::Half>()), static_cast<int>(n)},
        {reinterpret_cast<Element*>(out.data_ptr<at::Half>()), static_cast<int>(n)},
        {1.0f, 0.0f});

    Gemm gemm;
    TORCH_CHECK(gemm.can_implement(arguments) == cutlass::Status::kSuccess,
                "CUTLASS reference cannot implement this shape/alignment");
    TORCH_CHECK(gemm.initialize(arguments, nullptr, stream) == cutlass::Status::kSuccess,
                "CUTLASS reference initialization failed");
    TORCH_CHECK(gemm(stream) == cutlass::Status::kSuccess,
                "CUTLASS reference launch failed");
    last_info = ReferenceBackendInfo{0, 0};
#else
    TORCH_CHECK(false, "CUTLASS support was not built");
#endif
}

ReferenceBackendInfo cutlass_backend_info() {
    return last_info;
}

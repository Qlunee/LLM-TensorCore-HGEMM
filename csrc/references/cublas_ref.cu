#include "reference.h"

#include <climits>
#include <memory>

#include <cublas_v2.h>

#include "../common/checks.cuh"

namespace {

class CublasContext {
public:
    explicit CublasContext(int device) : device_(device) {
        CUDA_CHECK(cudaSetDevice(device_));
        CUBLAS_CHECK(cublasCreate(&handle_));
        CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH));
    }

    ~CublasContext() {
        if (handle_ != nullptr) {
            cudaSetDevice(device_);
            cublasDestroy(handle_);
        }
    }

    cublasHandle_t handle() const { return handle_; }
    int device() const { return device_; }

private:
    int device_;
    cublasHandle_t handle_{nullptr};
};

thread_local std::unique_ptr<CublasContext> context;

CublasContext& get_context(int device) {
    if (!context || context->device() != device) {
        context = std::make_unique<CublasContext>(device);
    }
    return *context;
}

}  // namespace

void launch_cublas_reference(const torch::Tensor& a, const torch::Tensor& b,
                             torch::Tensor& out, cudaStream_t stream) {
    const int64_t m = a.size(0);
    const int64_t k = a.size(1);
    const int64_t n = b.size(1);
    TORCH_CHECK(m <= INT_MAX && n <= INT_MAX && k <= INT_MAX,
                "cuBLAS V0 adapter supports dimensions up to INT_MAX");

    auto& ctx = get_context(a.get_device());
    CUBLAS_CHECK(cublasSetStream(ctx.handle(), stream));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Row-major C=A*B is the same memory operation as column-major C^T=B^T*A^T.
    CUBLAS_CHECK(cublasGemmEx(
        ctx.handle(), CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
        &alpha,
        b.data_ptr<at::Half>(), CUDA_R_16F, static_cast<int>(n),
        a.data_ptr<at::Half>(), CUDA_R_16F, static_cast<int>(k),
        &beta,
        out.data_ptr<at::Half>(), CUDA_R_16F, static_cast<int>(n),
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

ReferenceBackendInfo cublas_backend_info() {
    return ReferenceBackendInfo{static_cast<int64_t>(CUBLAS_GEMM_DEFAULT_TENSOR_OP), 0};
}

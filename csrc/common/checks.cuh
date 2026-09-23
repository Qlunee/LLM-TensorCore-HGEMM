#pragma once

#include <sstream>
#include <stdexcept>

#include <cublas_v2.h>
#include <cuda_runtime.h>

inline const char* cublas_status_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        const cudaError_t status_ = (expr);                                     \
        if (status_ != cudaSuccess) {                                           \
            std::ostringstream message_;                                       \
            message_ << "CUDA error at " << __FILE__ << ':' << __LINE__        \
                     << ": " << cudaGetErrorString(status_);                   \
            throw std::runtime_error(message_.str());                           \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(expr)                                                      \
    do {                                                                        \
        const cublasStatus_t status_ = (expr);                                  \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                                 \
            std::ostringstream message_;                                       \
            message_ << "cuBLAS error at " << __FILE__ << ':' << __LINE__      \
                     << ": " << cublas_status_string(status_);                 \
            throw std::runtime_error(message_.str());                           \
        }                                                                       \
    } while (0)

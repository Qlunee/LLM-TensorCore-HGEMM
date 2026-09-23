#pragma once

#include <cstdint>

#include <cuda_runtime.h>
#include <torch/extension.h>

struct ReferenceBackendInfo {
    int64_t algorithm_id{-1};
    int64_t workspace_bytes{0};
};

void launch_cublas_reference(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream);

void launch_cublaslt_reference(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream);

void launch_cutlass_reference(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream);

ReferenceBackendInfo cublas_backend_info();
ReferenceBackendInfo cublaslt_backend_info();
ReferenceBackendInfo cutlass_backend_info();
bool cutlass_reference_available();

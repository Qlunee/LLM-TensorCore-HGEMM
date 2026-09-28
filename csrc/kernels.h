#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

void launch_cuda_naive(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream);

void launch_cuda_tiled(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream);

void launch_wmma_basic(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream);

void launch_wmma_tiled(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream);

void launch_mma_ptx_probe(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream);

void launch_ldmatrix_probe(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream);

void launch_mma_ptx(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream);

void launch_mma_padded(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream);

void launch_mma_vectorized(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream);

void launch_mma_double_buffer(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream);

void launch_mma_async(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream);

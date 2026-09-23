#include "hgemm.h"

HgemmProblem validate_hgemm_inputs(
    const torch::Tensor& a,
    const torch::Tensor& b) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.device() == b.device(), "a and b must be on the same CUDA device");
    TORCH_CHECK(a.scalar_type() == torch::kFloat16, "a must have dtype torch.float16");
    TORCH_CHECK(b.scalar_type() == torch::kFloat16, "b must have dtype torch.float16");
    TORCH_CHECK(a.dim() == 2, "a must be two-dimensional, got ", a.dim(), " dimensions");
    TORCH_CHECK(b.dim() == 2, "b must be two-dimensional, got ", b.dim(), " dimensions");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous row-major");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous row-major");
    TORCH_CHECK(a.size(0) > 0 && a.size(1) > 0 && b.size(1) > 0,
                "M, N, and K must all be positive");
    TORCH_CHECK(a.size(1) == b.size(0),
                "K dimensions must match: a.shape[1]=", a.size(1),
                ", b.shape[0]=", b.size(0));

    return HgemmProblem{a.size(0), b.size(1), a.size(1), a.get_device()};
}

/*
**`hgemm.cpp` 是 HGEMM 的核心调度层：
它校验输入输出张量是否合法，解析后端名（`auto` 默认路由到 cuBLASLt），
然后在正确的 CUDA 设备与 stream 上，把计算分发到自定义 kernel 或参考实现，
并对外提供 `has_cutlass()` 和 `backend_info()` 两个查询接口。**
*/
#include "hgemm.h"
#include "kernels.h"

#include <c10/cuda/CUDAGuard.h>

#include "common/cuda_utils.cuh"
#include "references/reference.h"

namespace {

// 校验输出张量：设备、dtype、维度、形状、连续性必须与问题匹配
void validate_output(const torch::Tensor& out, const HgemmProblem& problem,
                     const torch::Tensor& a) {
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(out.device() == a.device(), "out must be on the same CUDA device as a and b");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must have dtype torch.float16");
    TORCH_CHECK(out.dim() == 2, "out must be two-dimensional");
    TORCH_CHECK(out.size(0) == problem.m && out.size(1) == problem.n,
                "out must have shape [", problem.m, ", ", problem.n, "]");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous row-major");
}

// 解析后端名："auto" 默认路由到 cuBLASLt
std::string resolve_implementation(const std::string& implementation) {
    return implementation == "auto" ? "cublaslt" : implementation;
}

}  // namespace

// 分配输出张量并调用 hgemm_out
torch::Tensor hgemm(const torch::Tensor& a, const torch::Tensor& b,
                    const std::string& implementation,
                    const std::string& epilogue) {
    const auto problem = validate_hgemm_inputs(a, b);
    auto out = torch::empty({problem.m, problem.n}, a.options());
    hgemm_out(a, b, out, implementation, epilogue);
    return out;
}

// 核心入口：校验输入输出，按 implementation 分发到对应后端
void hgemm_out(const torch::Tensor& a, const torch::Tensor& b,
               torch::Tensor out, const std::string& implementation,
               const std::string& epilogue) {
    const auto problem = validate_hgemm_inputs(a, b);
    validate_output(out, problem, a);
    TORCH_CHECK(epilogue == "none", "HGEMM supports only epilogue='none'");

    const c10::cuda::CUDAGuard device_guard(a.device());  // 切换/锁定当前 CUDA 设备
    const auto stream = current_stream(problem.device);    // 获取当前 stream
    const auto resolved = resolve_implementation(implementation);

    if (resolved == "cuda_naive") {
        launch_cuda_naive(a, b, out, stream);
    } else if (resolved == "cuda_tiled") {
        launch_cuda_tiled(a, b, out, stream);
    } else if (resolved == "wmma_basic") {
        launch_wmma_basic(a, b, out, stream);
    } else if (resolved == "wmma_tiled") {
        launch_wmma_tiled(a, b, out, stream);
    } else if (resolved == "mma_ptx_probe") {
        launch_mma_ptx_probe(a, b, out, stream);
    } else if (resolved == "ldmatrix_probe") {
        launch_ldmatrix_probe(a, b, out, stream);
    } else if (resolved == "mma_ptx") {
        launch_mma_ptx(a, b, out, stream);
    } else if (resolved == "mma_padded") {
        launch_mma_padded(a, b, out, stream);
    } else if (resolved == "mma_vectorized") {
        launch_mma_vectorized(a, b, out, stream);
    } else if (resolved == "mma_double_buffer") {
        launch_mma_double_buffer(a, b, out, stream);
    } else if (resolved == "mma_async") {
        launch_mma_async(a, b, out, stream);
    } else if (resolved == "mma_double_buffer_compact") {
        launch_mma_double_buffer_compact(a, b, out, stream);
    } else if (resolved == "mma_async_compact") {
        launch_mma_async_compact(a, b, out, stream);
    } else if (resolved == "cublas") {
        launch_cublas_reference(a, b, out, stream);
    } else if (resolved == "cublaslt") {
        launch_cublaslt_reference(a, b, out, stream);
    } else if (resolved == "cutlass") {
        // CUTLASS 为可选编译：未构建时直接报错
        TORCH_CHECK(cutlass_reference_available(),
                    "CUTLASS support was not built. Set CUTLASS_HOME and reinstall.");
        launch_cutlass_reference(a, b, out, stream);
    } else {
        TORCH_CHECK(false, "unknown implementation: ", implementation);
    }
}

// 查询当前构建是否包含 CUTLASS 支持
bool has_cutlass() {
    return cutlass_reference_available();
}

// 返回指定后端的元信息（算法 ID、工作空间大小）
std::map<std::string, int64_t> backend_info(const std::string& implementation) {
    const auto resolved = resolve_implementation(implementation);
    if (resolved == "cuda_naive" || resolved == "cuda_tiled" ||
        resolved == "wmma_basic" || resolved == "wmma_tiled" ||
        resolved == "mma_ptx_probe" || resolved == "ldmatrix_probe" ||
        resolved == "mma_ptx" || resolved == "mma_padded" ||
        resolved == "mma_vectorized" || resolved == "mma_double_buffer" ||
        resolved == "mma_async" || resolved == "mma_double_buffer_compact" ||
        resolved == "mma_async_compact") {
        return {{"algorithm_id", -1}, {"workspace_bytes", 0}};
    }

    ReferenceBackendInfo info;
    if (resolved == "cublas") {
        info = cublas_backend_info();
    } else if (resolved == "cublaslt") {
        info = cublaslt_backend_info();
    } else if (resolved == "cutlass") {
        info = cutlass_backend_info();
    } else {
        TORCH_CHECK(false, "unknown implementation: ", implementation);
    }
    return {{"algorithm_id", info.algorithm_id},
            {"workspace_bytes", info.workspace_bytes}};
}

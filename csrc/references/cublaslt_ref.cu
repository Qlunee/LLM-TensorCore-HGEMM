#include "reference.h"

#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

#include <cublasLt.h>

#include "../common/checks.cuh"

namespace {

constexpr size_t kWorkspaceBytes = 32ULL * 1024ULL * 1024ULL;

struct PlanKey {
    int64_t m;
    int64_t n;
    int64_t k;

    bool operator==(const PlanKey& other) const {
        return m == other.m && n == other.n && k == other.k;
    }
};

struct PlanKeyHash {
    size_t operator()(const PlanKey& key) const {
        size_t value = std::hash<int64_t>{}(key.m);
        value ^= std::hash<int64_t>{}(key.n) + 0x9e3779b9 + (value << 6) + (value >> 2);
        value ^= std::hash<int64_t>{}(key.k) + 0x9e3779b9 + (value << 6) + (value >> 2);
        return value;
    }
};

class LtPlan {
public:
    LtPlan(cublasLtHandle_t handle, int64_t m, int64_t n, int64_t k) {
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32F, CUDA_R_32F));

        const cublasOperation_t no_transpose = CUBLAS_OP_N;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            operation_, CUBLASLT_MATMUL_DESC_TRANSA,
            &no_transpose, sizeof(no_transpose)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            operation_, CUBLASLT_MATMUL_DESC_TRANSB,
            &no_transpose, sizeof(no_transpose)));

        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a_layout_, CUDA_R_16F, m, k, k));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b_layout_, CUDA_R_16F, k, n, n));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c_layout_, CUDA_R_16F, m, n, n));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&d_layout_, CUDA_R_16F, m, n, n));

        const cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
        for (auto layout : {a_layout_, b_layout_, c_layout_, d_layout_}) {
            CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
                layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                &row_order, sizeof(row_order)));
        }

        cublasLtMatmulPreference_t preference = nullptr;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
        try {
            CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
                preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &kWorkspaceBytes, sizeof(kWorkspaceBytes)));

            std::array<cublasLtMatmulHeuristicResult_t, 16> candidates{};
            int returned = 0;
            CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
                handle, operation_, a_layout_, b_layout_, c_layout_, d_layout_,
                preference, static_cast<int>(candidates.size()), candidates.data(), &returned));
            TORCH_CHECK(returned > 0, "cuBLASLt returned no heuristic algorithm for ",
                        m, "x", n, "x", k);

            bool found = false;
            for (int i = 0; i < returned; ++i) {
                if (candidates[i].state == CUBLAS_STATUS_SUCCESS &&
                    candidates[i].workspaceSize <= kWorkspaceBytes) {
                    algorithm_ = candidates[i].algo;
                    workspace_bytes_ = candidates[i].workspaceSize;
                    found = true;
                    break;
                }
            }
            TORCH_CHECK(found, "cuBLASLt returned no successful heuristic algorithm");
        } catch (...) {
            cublasLtMatmulPreferenceDestroy(preference);
            throw;
        }
        CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));

        size_t written = 0;
        CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
            &algorithm_, CUBLASLT_ALGO_CONFIG_ID,
            &algorithm_id_, sizeof(algorithm_id_), &written));
    }

    ~LtPlan() {
        if (d_layout_) cublasLtMatrixLayoutDestroy(d_layout_);
        if (c_layout_) cublasLtMatrixLayoutDestroy(c_layout_);
        if (b_layout_) cublasLtMatrixLayoutDestroy(b_layout_);
        if (a_layout_) cublasLtMatrixLayoutDestroy(a_layout_);
        if (operation_) cublasLtMatmulDescDestroy(operation_);
    }

    LtPlan(const LtPlan&) = delete;
    LtPlan& operator=(const LtPlan&) = delete;

    cublasLtMatmulDesc_t operation() const { return operation_; }
    cublasLtMatrixLayout_t a_layout() const { return a_layout_; }
    cublasLtMatrixLayout_t b_layout() const { return b_layout_; }
    cublasLtMatrixLayout_t c_layout() const { return c_layout_; }
    cublasLtMatrixLayout_t d_layout() const { return d_layout_; }
    const cublasLtMatmulAlgo_t* algorithm() const { return &algorithm_; }
    size_t workspace_bytes() const { return workspace_bytes_; }
    int algorithm_id() const { return algorithm_id_; }

private:
    cublasLtMatmulDesc_t operation_{nullptr};
    cublasLtMatrixLayout_t a_layout_{nullptr};
    cublasLtMatrixLayout_t b_layout_{nullptr};
    cublasLtMatrixLayout_t c_layout_{nullptr};
    cublasLtMatrixLayout_t d_layout_{nullptr};
    cublasLtMatmulAlgo_t algorithm_{};
    size_t workspace_bytes_{0};
    int algorithm_id_{-1};
};

class LtContext {
public:
    explicit LtContext(int device) : device_(device) {
        CUDA_CHECK(cudaSetDevice(device_));
        CUBLAS_CHECK(cublasLtCreate(&handle_));
        CUDA_CHECK(cudaMalloc(&workspace_, kWorkspaceBytes));
    }

    ~LtContext() {
        plans_.clear();
        if (workspace_) {
            cudaSetDevice(device_);
            cudaFree(workspace_);
        }
        if (handle_) cublasLtDestroy(handle_);
    }

    int device() const { return device_; }
    cublasLtHandle_t handle() const { return handle_; }
    void* workspace() const { return workspace_; }

    LtPlan& plan(int64_t m, int64_t n, int64_t k) {
        const PlanKey key{m, n, k};
        auto found = plans_.find(key);
        if (found == plans_.end()) {
            found = plans_.emplace(
                key, std::make_unique<LtPlan>(handle_, m, n, k)).first;
        }
        return *found->second;
    }

private:
    int device_;
    cublasLtHandle_t handle_{nullptr};
    void* workspace_{nullptr};
    std::unordered_map<PlanKey, std::unique_ptr<LtPlan>, PlanKeyHash> plans_;
};

thread_local std::unique_ptr<LtContext> context;
thread_local ReferenceBackendInfo last_info;

LtContext& get_context(int device) {
    if (!context || context->device() != device) {
        context = std::make_unique<LtContext>(device);
    }
    return *context;
}

}  // namespace

void launch_cublaslt_reference(const torch::Tensor& a, const torch::Tensor& b,
                               torch::Tensor& out, cudaStream_t stream) {
    const int64_t m = a.size(0);
    const int64_t k = a.size(1);
    const int64_t n = b.size(1);
    TORCH_CHECK(m <= INT_MAX && n <= INT_MAX && k <= INT_MAX,
                "cuBLASLt V0 adapter supports dimensions up to INT_MAX");

    auto& ctx = get_context(a.get_device());
    auto& plan = ctx.plan(m, n, k);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasLtMatmul(
        ctx.handle(), plan.operation(), &alpha,
        a.data_ptr<at::Half>(), plan.a_layout(),
        b.data_ptr<at::Half>(), plan.b_layout(),
        &beta,
        out.data_ptr<at::Half>(), plan.c_layout(),
        out.data_ptr<at::Half>(), plan.d_layout(),
        plan.algorithm(), ctx.workspace(), plan.workspace_bytes(), stream));

    last_info = ReferenceBackendInfo{
        static_cast<int64_t>(plan.algorithm_id()),
        static_cast<int64_t>(plan.workspace_bytes())};
}

ReferenceBackendInfo cublaslt_backend_info() {
    return last_info;
}

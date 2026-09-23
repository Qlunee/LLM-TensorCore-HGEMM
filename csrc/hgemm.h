#pragma once

#include <cstdint>
#include <map>
#include <string>

#include <torch/extension.h>

struct HgemmProblem {
    int64_t m;
    int64_t n;
    int64_t k;
    int device;
};

HgemmProblem validate_hgemm_inputs(
    const torch::Tensor& a,
    const torch::Tensor& b);

torch::Tensor hgemm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const std::string& implementation,
    const std::string& epilogue);

void hgemm_out(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor out,
    const std::string& implementation,
    const std::string& epilogue);

bool has_cutlass();
std::map<std::string, int64_t> backend_info(const std::string& implementation);

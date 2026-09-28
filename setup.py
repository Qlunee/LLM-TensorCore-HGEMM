from __future__ import annotations

import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent.resolve()
CUTLASS_HOME = os.environ.get("CUTLASS_HOME")

include_dirs = [str(ROOT / "csrc")]
nvcc_flags = [
    "-O3",
    "-std=c++17",
    "-lineinfo",
    "-gencode=arch=compute_86,code=sm_86",
]

if CUTLASS_HOME:
    cutlass_include = Path(CUTLASS_HOME).expanduser().resolve() / "include"
    if not (cutlass_include / "cutlass" / "cutlass.h").exists():
        raise RuntimeError(f"CUTLASS_HOME does not contain include/cutlass/cutlass.h: {CUTLASS_HOME}")
    include_dirs.append(str(cutlass_include))
    nvcc_flags.append("-DLLM_HGEMM_WITH_CUTLASS=1")

extension = CUDAExtension(
    name="llm_hgemm._C",
    sources=[
        "csrc/bindings.cpp",
        "csrc/contract.cpp",
        "csrc/dispatch.cpp",
        "csrc/v1_cuda_naive.cu",
        "csrc/v1_cuda_tiled.cu",
        "csrc/v2_wmma_basic.cu",
        "csrc/v3_wmma_tiled.cu",
        "csrc/v4_mma_ptx.cu",
        "csrc/v5_mma_vectorized.cu",
        "csrc/references/cublas_ref.cu",
        "csrc/references/cublaslt_ref.cu",
        "csrc/references/cutlass_ref.cu",
    ],
    include_dirs=include_dirs,
    libraries=["cublas", "cublasLt"],
    extra_compile_args={
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": nvcc_flags,
    },
)

setup(
    name="llm-tensorcore-hgemm",
    version="0.1.0",
    description="Ampere Tensor Core HGEMM kernels for LLM linear shapes",
    packages=find_packages(),
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    python_requires=">=3.10",
)

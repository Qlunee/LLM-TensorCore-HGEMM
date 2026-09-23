#pragma once

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

inline cudaStream_t current_stream(int device) {
    return c10::cuda::getCurrentCUDAStream(device).stream();
}

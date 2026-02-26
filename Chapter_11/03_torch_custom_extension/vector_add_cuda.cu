/*
 * vector_add_cuda.cu
 * CUDA kernel for fused vector_add_scale: z = scale * (a + b)
 * Called from the PyTorch C++ extension wrapper (vector_add.cpp).
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel: z[i] = scale * (a[i] + b[i])
__global__ void vector_add_scale_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ z,
    float scale,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        z[i] = scale * (a[i] + b[i]);
}

// Called from vector_add.cpp
torch::Tensor vector_add_scale_cuda(
    torch::Tensor a,
    torch::Tensor b,
    float scale)
{
    TORCH_CHECK(a.is_cuda(), "a must be on GPU");
    TORCH_CHECK(b.is_cuda(), "b must be on GPU");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have same shape");

    auto z = torch::zeros_like(a);
    int n  = a.numel();
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    vector_add_scale_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        z.data_ptr<float>(),
        scale,
        n
    );

    return z;
}

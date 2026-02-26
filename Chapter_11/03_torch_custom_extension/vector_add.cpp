/*
 * vector_add.cpp
 * C++ wrapper that binds the CUDA kernel to Python via PyTorch's pybind11.
 */

#include <torch/extension.h>

// Forward declaration (defined in vector_add_cuda.cu)
torch::Tensor vector_add_scale_cuda(
    torch::Tensor a,
    torch::Tensor b,
    float scale);

// Python-facing function: validates inputs, dispatches to CUDA
torch::Tensor vector_add_scale(
    torch::Tensor a,
    torch::Tensor b,
    float scale)
{
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

    return vector_add_scale_cuda(a, b, scale);
}

// PYBIND11_MODULE creates the Python extension module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add_scale", &vector_add_scale,
          "Fused vector add+scale: z = scale * (a + b)",
          py::arg("a"), py::arg("b"), py::arg("scale") = 1.0f);
}

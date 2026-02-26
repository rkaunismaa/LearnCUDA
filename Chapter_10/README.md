# Chapter 10: CUDA Libraries — cuBLAS, Thrust, and cuRAND

## 10.1 Why Use Libraries?

NVIDIA provides battle-hardened libraries that exploit every GPU optimization:
- Hardware-specific kernel dispatch (different code for each GPU generation)
- Tensor Core usage for matrix operations
- Auto-tuned parameters for different matrix sizes
- Continuous updates as new hardware ships

For most production workloads, **use libraries first, write custom kernels only when needed**.

## 10.2 cuBLAS

cuBLAS is NVIDIA's implementation of the BLAS (Basic Linear Algebra Subprograms) standard.

### BLAS Levels
- **Level 1**: Vector-vector operations (dot product, axpy, norms)
- **Level 2**: Matrix-vector operations (gemv)
- **Level 3**: Matrix-matrix operations (gemm)

### Column-Major Convention (Critical!)

cuBLAS uses **column-major** storage (like Fortran), but C/C++ uses **row-major**. This is the #1 source of confusion.

In column-major, element `[row][col]` of a matrix of dimensions `(rows, cols)` is stored at index `col * rows + row`.

**Trick to use row-major arrays with cuBLAS**: use the identity `(A·B)ᵀ = Bᵀ·Aᵀ`:

```c
// We have C (M×N) = A (M×K) × B (K×N), row-major
// cuBLAS computes C_col = A_col × B_col (column-major)
//
// Row-major A of shape (M,K) = Column-major Aᵀ of shape (K,M)
// So compute: C_colmaj (N×M) = B_colmaj (N×K) × A_colmaj (K×M)
// Which in cuBLAS is: cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
//                                 N, M, K, &alpha, B, N, A, K, &beta, C, N)
// The result C_colmaj is exactly C_rowmaj transposed — which is C!
```

### cublasSgemm Signature

```c
cublasStatus_t cublasSgemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,    // Scalar for A*B
    const float *A, int lda,   // Leading dimension of A
    const float *B, int ldb,   // Leading dimension of B
    const float *beta,     // Scalar for C (accumulate)
    float *C, int ldc          // Leading dimension of C
);
// Computes: C = alpha * op(A) * op(B) + beta * C
```

## 10.3 Thrust

Thrust is a C++ template library that provides STL-like algorithms for GPU:

```cpp
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

thrust::device_vector<float> d_vec(1000, 1.0f);  // GPU vector of 1000 ones

// Transform: square each element
thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(),
                  [] __device__ (float x) { return x * x; });

// Reduce: sum
float sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f);

// Sort
thrust::sort(d_vec.begin(), d_vec.end());

// Transfer to/from host
thrust::host_vector<float> h_vec = d_vec;  // Implicit D2H copy
```

Thrust is **header-only** — no linking required.

## 10.4 cuRAND

cuRAND generates random numbers on the GPU:

```c
#include <curand.h>

curandGenerator_t gen;
curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
curandSetPseudoRandomGeneratorSeed(gen, 42);

float *d_randoms;
cudaMalloc(&d_randoms, n * sizeof(float));

// Generate n uniform floats in [0, 1)
curandGenerateUniform(gen, d_randoms, n);

// Generate n normally-distributed floats (mean=0, stddev=1)
curandGenerateNormal(gen, d_randoms, n, 0.0f, 1.0f);

curandDestroyGenerator(gen);
```

Link with `-lcurand`.

## 10.5 Other Libraries

| Library | Purpose |
|---------|---------|
| cuFFT | Fast Fourier Transform |
| cuSPARSE | Sparse matrix operations |
| cuDNN | Deep neural network primitives |
| cuSolver | Dense/sparse linear system solvers |
| NCCL | Multi-GPU collective operations |
| cub | Low-level GPU primitives (building blocks for libraries) |

## 10.6 Exercises

1. In `01_cublas_gemm.cu`, change the matrix size from 2048 to 512 and 4096. How does GFLOPS scale with matrix size? (Hint: cuBLAS has overhead for small matrices.)
2. Implement a batched GEMM using `cublasSgemmBatched` for 100 small 64×64 matrices.
3. In `02_thrust_basics.cu`, use `thrust::transform_reduce` to compute the L2 norm (sqrt of sum of squares) in a single call.
4. Write a Thrust-based filter that extracts only positive elements from a vector using `thrust::copy_if`.
5. Use cuRAND to verify the Central Limit Theorem: generate 1000 samples of 100 uniform random numbers each, compute their means, and verify the mean of means ≈ 0.5 and stddev of means ≈ 0.5/√100.

## 10.7 Key Takeaways

- cuBLAS uses **column-major** storage — use the transpose trick for row-major arrays.
- cuBLAS GEMM is **orders of magnitude faster** than our hand-written tiled version (Tensor Cores, auto-tuning, etc.).
- Thrust provides STL-like algorithms; use it for quick GPU algorithm prototyping.
- cuRAND generates high-quality random numbers efficiently on the GPU.
- Use libraries as building blocks; write custom kernels only for unique workloads.

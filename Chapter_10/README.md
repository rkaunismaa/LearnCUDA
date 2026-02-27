# Chapter 10: CUDA Libraries — cuBLAS, Thrust, and cuRAND

## 10.1 Why Use Libraries?

NVIDIA provides battle-hardened libraries that exploit every GPU optimization: hardware-specific kernel dispatch, Tensor Core usage, auto-tuned parameters, and continuous updates as new hardware ships.

```mermaid
graph TD
    subgraph PROBLEM["The Problem: Reinventing the Wheel"]
        HAND["Hand-written GEMM\\n(Chapter 4 tiled kernel)\\n~5 TFLOPS peak"]
        MANUAL["Manual random gen\\nNaive FFT\\nCustom sort"]
    end

    subgraph SOLUTION["CUDA Libraries — Already Optimized"]
        CB["cuBLAS\\nMatrix algebra\\nTensor Cores → 100+ TFLOPS"]
        TH["Thrust\\nSTL-like algorithms\\nsort, reduce, transform"]
        CR["cuRAND\\nGPU random numbers\\nhigh-quality PRNG"]
        OT["cuFFT / cuDNN\\ncuSPARSE / NCCL\\ncub / cuSolver"]
    end

    HAND -->|"replace with"| CB
    MANUAL -->|"replace with"| TH
    MANUAL -->|"replace with"| CR
    MANUAL -->|"replace with"| OT

    style PROBLEM  fill:#2a0d0d,color:#f1948a,stroke:#e74c3c
    style SOLUTION fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style HAND     fill:#c0392b,color:#fff,stroke:#922b21
    style MANUAL   fill:#c0392b,color:#fff,stroke:#922b21
    style CB       fill:#1e8449,color:#fff,stroke:#196f3d
    style TH       fill:#1e8449,color:#fff,stroke:#196f3d
    style CR       fill:#1e8449,color:#fff,stroke:#196f3d
    style OT       fill:#1f618d,color:#fff,stroke:#154360
```

**Rule**: Use libraries first, write custom kernels only for unique workloads that no library covers.

## 10.2 cuBLAS

cuBLAS is NVIDIA's GPU implementation of the BLAS (Basic Linear Algebra Subprograms) standard.

### BLAS Level Hierarchy

```mermaid
graph TD
    subgraph BLAS["BLAS Levels — Increasing Arithmetic Intensity"]
        L1["Level 1 — Vector × Vector\\ndot(x, y)         → scalar\\naxpy(α, x, y)     → vector y += αx\\nnrm2(x)           → scalar\\nO(N) work, O(N) data → memory-bound"]
        L2["Level 2 — Matrix × Vector\\ngemv(A, x)        → y = αAx + βy\\nA is M×N, x is N, y is M\\nO(MN) work, O(MN) data → memory-bound"]
        L3["Level 3 — Matrix × Matrix\\ngemm(A, B)        → C = αAB + βC\\nA: M×K, B: K×N, C: M×N\\nO(MNK) work, O(MN+MK+KN) data\\n→ compute-bound for large M,N,K ✓ (Tensor Cores shine here)"]
    end

    L1 --> L2 --> L3

    style BLAS fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style L1   fill:#c0392b,color:#fff,stroke:#922b21
    style L2   fill:#d35400,color:#fff,stroke:#a04000
    style L3   fill:#1e8449,color:#fff,stroke:#196f3d
```

### Column-Major Convention (Critical!)

cuBLAS uses **column-major** storage (like Fortran), but C/C++ uses **row-major**. This is the #1 source of confusion.

```
Row-major (C/C++):          Column-major (cuBLAS/Fortran):
Matrix A =                  Matrix A =
  [1  2  3]                   [1  2  3]
  [4  5  6]                   [4  5  6]

  Memory layout (row-major):   Memory layout (col-major):
  [1, 2, 3, 4, 5, 6]          [1, 4, 2, 5, 3, 6]
   row 0──────  row 1──────    col 0──  col 1──  col 2──

  element [r][c] at index:     element [r][c] at index:
    c + r * num_cols              r + c * num_rows
```

```diff
  Naive approach — will produce WRONG results:

- // Row-major C array passed directly to cuBLAS
- cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
-             &alpha, A, M, B, K, &beta, C, M);
- // cuBLAS interprets row-major A as column-major Aᵀ → wrong answer ✗

  Correct approach — use the identity (A·B)ᵀ = Bᵀ·Aᵀ:

+ // Row-major A(M×K) = Column-major Aᵀ(K×M)
+ // We want C = A·B, so compute Cᵀ = Bᵀ·Aᵀ
+ // In column-major terms: C_out(N×M) = B_in(N×K) · A_in(K×M)
+ cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
+             N, M, K,              // swap N and M
+             &alpha, B, N, A, K,   // swap A and B, use N as lda
+             &beta, C, N);         // ldc = N
+ // The result stored in C is our correct row-major C ✓
```

### cublasSgemm Call Flow

```mermaid
sequenceDiagram
    participant CPU  as 🖥️ Host
    participant H    as cuBLAS Handle
    participant GPU  as 🎮 Device

    CPU  ->> H:   cublasCreate(&handle)
    Note over H: handle manages cuBLAS state & stream

    CPU  ->> GPU: cudaMalloc d_A, d_B, d_C
    CPU  ->> GPU: cudaMemcpy A → d_A, B → d_B

    CPU  ->> H:   cublasSgemm(handle,<br/>CUBLAS_OP_N, CUBLAS_OP_N,<br/>N, M, K,<br/>&alpha, d_B, N, d_A, K,<br/>&beta, d_C, N)
    Note over H: OP_N = no transpose<br/>alpha=1, beta=0 for plain multiply
    H   ->> GPU: dispatch optimized GEMM kernel
    Note over GPU: Tensor Cores compute<br/>C = α·op(A)·op(B) + β·C

    CPU  ->> GPU: cudaMemcpy d_C → C
    CPU  ->> H:   cublasDestroy(handle)
```

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

### cuBLAS vs Hand-Written Tiled GEMM

```mermaid
graph LR
    subgraph CUSTOM["Chapter 4 Tiled GEMM"]
        C1["Fixed 32×32 tile size"]
        C2["No Tensor Cores"]
        C3["~1–5 TFLOPS"]
        C4["Single SM arch"]
    end

    subgraph CUBLAS["cuBLAS GEMM"]
        B1["Auto-tuned tile sizes\\nper GPU architecture"]
        B2["Tensor Core WMMA\\n(FP16 → FP32 accumulate)"]
        B3["100–300+ TFLOPS\\n(on RTX 4090)"]
        B4["Hardware dispatch table\\nsm_75, sm_80, sm_89..."]
    end

    CUSTOM -->|"replace with"| CUBLAS

    style CUSTOM fill:#2a0d0d,color:#f1948a,stroke:#e74c3c
    style CUBLAS fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style C1 fill:#c0392b,color:#fff,stroke:#922b21
    style C2 fill:#c0392b,color:#fff,stroke:#922b21
    style C3 fill:#c0392b,color:#fff,stroke:#922b21
    style C4 fill:#c0392b,color:#fff,stroke:#922b21
    style B1 fill:#1e8449,color:#fff,stroke:#196f3d
    style B2 fill:#1e8449,color:#fff,stroke:#196f3d
    style B3 fill:#1e8449,color:#fff,stroke:#196f3d
    style B4 fill:#1e8449,color:#fff,stroke:#196f3d
```

## 10.3 Thrust

Thrust is a C++ template library that provides STL-like algorithms for GPU. It is **header-only** — no linking required.

```mermaid
graph LR
    subgraph PIPELINE["Thrust Algorithm Pipeline"]
        HV["thrust::host_vector\\n(RAM — CPU-side)"]
        DV["thrust::device_vector\\n(VRAM — GPU-side)"]
        TR["thrust::transform\\nper-element map\\nλ: x → f(x)"]
        RE["thrust::reduce\\nfold to scalar\\nsum / min / max"]
        SO["thrust::sort\\nparallel radix sort"]
        FI["thrust::copy_if\\nfilter by predicate\\nλ: x → bool"]
    end

    HV -->|"implicit H2D copy\\nassignment"| DV
    DV --> TR
    TR --> RE
    DV --> SO
    DV --> FI
    DV -->|"implicit D2H copy\\nassignment"| HV

    style PIPELINE fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style HV fill:#1f618d,color:#fff,stroke:#154360
    style DV fill:#1e8449,color:#fff,stroke:#196f3d
    style TR fill:#7d3c98,color:#fff,stroke:#6c3483
    style RE fill:#d35400,color:#fff,stroke:#a04000
    style SO fill:#1f618d,color:#fff,stroke:#154360
    style FI fill:#c0392b,color:#fff,stroke:#922b21
```

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

// Filter: keep only positive elements
thrust::device_vector<float> d_pos(d_vec.size());
auto end = thrust::copy_if(d_vec.begin(), d_vec.end(), d_pos.begin(),
                           [] __device__ (float x) { return x > 0.0f; });

// Transfer to/from host
thrust::host_vector<float> h_vec = d_vec;  // Implicit D2H copy
```

```diff
  Thrust vs raw CUDA — trade-offs:

+ Thrust: simple, composable, header-only, works like STL       ✓
+ Thrust: handles memory management automatically               ✓
+ Thrust: good for prototyping and non-critical paths           ✓
- Thrust: less control over shared memory / block size          ✗
- Thrust: harder to fuse multiple passes into one kernel        ✗
- Thrust: custom operations must be device lambdas              ✗

  Use Thrust when correctness and development speed matter more than
  squeezing out the last 10% of performance.
```

## 10.4 cuRAND

cuRAND generates high-quality random numbers directly on the GPU, avoiding the PCIe bottleneck of CPU-generated random data.

```mermaid
sequenceDiagram
    participant CPU as 🖥️ Host
    participant GEN as curandGenerator
    participant GPU as 🎮 Device

    CPU  ->> GEN: curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)
    CPU  ->> GEN: curandSetPseudoRandomGeneratorSeed(gen, 42)
    Note over GEN: CURAND_RNG_PSEUDO_DEFAULT = XORWOW<br/>Other options: MRG32k3a, MTGP32,<br/>SOBOL (quasi-random)

    CPU  ->> GPU: cudaMalloc(&d_randoms, n * sizeof(float))

    CPU  ->> GEN: curandGenerateUniform(gen, d_randoms, n)
    GEN ->> GPU: launch PRNG kernel
    Note over GPU: n uniform floats in [0.0, 1.0)<br/>generated in parallel across GPU

    CPU  ->> GEN: curandGenerateNormal(gen, d_randoms, n, 0.0f, 1.0f)
    GEN ->> GPU: launch Box-Muller kernel
    Note over GPU: n normal floats (μ=0, σ=1)<br/>Box-Muller transform applied on GPU

    CPU  ->> GEN: curandDestroyGenerator(gen)
```

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

```mermaid
graph TD
    subgraph SIGNAL["Signal & Transform"]
        FFT["cuFFT\\nFast Fourier Transform\\nfft, ifft, rfft\\n-lcufft"]
    end

    subgraph SPARSE["Sparse Linear Algebra"]
        SP["cuSPARSE\\nSparse matrix ops\\nSpMM, SpMV, SpGEMM\\n-lcusparse"]
        SO["cuSolver\\nLinear system solvers\\nLU, QR, SVD, eigenvalues\\n-lcusolver"]
    end

    subgraph DL["Deep Learning"]
        DN["cuDNN\\nConv, BN, RNN, attention\\nUsed by PyTorch/TF\\n-lcudnn"]
    end

    subgraph LOWLEVEL["Low-Level Primitives"]
        CUB["cub\\nBlock/warp/device primitives\\nreduce, scan, sort, histogram\\nheader-only (ships with CUDA)"]
    end

    subgraph MULTI["Multi-GPU"]
        NC["NCCL\\nCollective operations\\nAllReduce, Broadcast, Gather\\nused by distributed training\\n-lnccl"]
    end

    style SIGNAL   fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style SPARSE   fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style DL       fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style LOWLEVEL fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style MULTI    fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style FFT fill:#7d3c98,color:#fff,stroke:#6c3483
    style SP  fill:#d35400,color:#fff,stroke:#a04000
    style SO  fill:#d35400,color:#fff,stroke:#a04000
    style DN  fill:#1e8449,color:#fff,stroke:#196f3d
    style CUB fill:#1f618d,color:#fff,stroke:#154360
    style NC  fill:#c0392b,color:#fff,stroke:#922b21
```

| Library | Purpose | Link Flag |
|---------|---------|-----------|
| cuFFT | Fast Fourier Transform | `-lcufft` |
| cuSPARSE | Sparse matrix operations | `-lcusparse` |
| cuDNN | Deep neural network primitives | `-lcudnn` |
| cuSolver | Dense/sparse linear system solvers | `-lcusolver` |
| NCCL | Multi-GPU collective operations | `-lnccl` |
| cub | Low-level GPU primitives (building blocks) | header-only |

### When to Use Which Library

```mermaid
flowchart TD
    START["What operation do you need?"]

    START --> Q1{"Matrix multiply\\nor linear algebra?"}
    Q1 -->|Dense| CUBLAS["cuBLAS\\ncublasSgemm, cublasSgemv"]
    Q1 -->|Sparse| CUSPARSE["cuSPARSE\\ncusparseSpMM, cusparseSpMV"]
    Q1 -->|Solve Ax=b| CUSOLVER["cuSolver\\ncusolverDnSgesv"]

    START --> Q2{"FFT / frequency\\ndomain?"}
    Q2 -->|Yes| CUFFT["cuFFT\\ncufftExecC2C"]

    START --> Q3{"Random numbers?"}
    Q3 -->|Yes| CURAND["cuRAND\\ncurandGenerateUniform"]

    START --> Q4{"STL-like GPU\\nalgorithm?"}
    Q4 -->|sort/reduce/transform| THRUST["Thrust"]
    Q4 -->|"building block for\\nyour own library"| CUB_["cub::DeviceReduce\\ncub::DeviceScan"]

    START --> Q5{"Multi-GPU\\ncollective?"}
    Q5 -->|AllReduce / Scatter| NCCL_["NCCL\\nncclAllReduce"]

    START --> Q6{"Deep learning\\nprimitive?"}
    Q6 -->|Conv/BN/Attn| CUDNN["cuDNN\\ncudnnConvolutionForward"]

    style START    fill:#7d3c98,color:#fff,stroke:#6c3483
    style Q1       fill:#1f618d,color:#fff,stroke:#154360
    style Q2       fill:#1f618d,color:#fff,stroke:#154360
    style Q3       fill:#1f618d,color:#fff,stroke:#154360
    style Q4       fill:#1f618d,color:#fff,stroke:#154360
    style Q5       fill:#1f618d,color:#fff,stroke:#154360
    style Q6       fill:#1f618d,color:#fff,stroke:#154360
    style CUBLAS   fill:#1e8449,color:#fff,stroke:#196f3d
    style CUSPARSE fill:#1e8449,color:#fff,stroke:#196f3d
    style CUSOLVER fill:#1e8449,color:#fff,stroke:#196f3d
    style CUFFT    fill:#1e8449,color:#fff,stroke:#196f3d
    style CURAND   fill:#1e8449,color:#fff,stroke:#196f3d
    style THRUST   fill:#1e8449,color:#fff,stroke:#196f3d
    style CUB_     fill:#d35400,color:#fff,stroke:#a04000
    style NCCL_    fill:#1e8449,color:#fff,stroke:#196f3d
    style CUDNN    fill:#1e8449,color:#fff,stroke:#196f3d
```

## 10.6 Exercises

1. In `01_cublas_gemm.cu`, change the matrix size from 2048 to 512 and 4096. How does GFLOPS scale with matrix size? (Hint: cuBLAS has overhead for small matrices.)
2. Implement a batched GEMM using `cublasSgemmBatched` for 100 small 64×64 matrices.
3. In `02_thrust_basics.cu`, use `thrust::transform_reduce` to compute the L2 norm (sqrt of sum of squares) in a single call.
4. Write a Thrust-based filter that extracts only positive elements from a vector using `thrust::copy_if`.
5. Use cuRAND to verify the Central Limit Theorem: generate 1000 samples of 100 uniform random numbers each, compute their means, and verify the mean of means ≈ 0.5 and stddev of means ≈ 0.5/√100.

## 10.7 Key Takeaways

- cuBLAS uses **column-major** storage — use the transpose trick `(A·B)ᵀ = Bᵀ·Aᵀ` for row-major arrays.
- cuBLAS GEMM is **orders of magnitude faster** than our hand-written tiled version (Tensor Cores, auto-tuning, etc.).
- Thrust provides STL-like algorithms; use it for quick GPU algorithm prototyping.
- cuRAND generates high-quality random numbers efficiently on the GPU — avoids PCIe transfer of CPU-generated data.
- `cub` provides the low-level building blocks that Thrust and cuBLAS are built on — use it when you need fine-grained control.
- Use libraries as building blocks; write custom kernels only for unique workloads.

/*
 * Chapter 12 — 02_vectorized_loads.cu
 *
 * Compares scalar (float) vs vectorized (float4) memory access.
 * float4 loads 16 bytes per instruction instead of 4.
 *
 * Also demonstrates __ldg() for read-only cache access and
 * instruction-level parallelism (ILP) via multiple accumulators.
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o vec_loads 02_vectorized_loads.cu
 * Run:
 *   ./vec_loads
 */

#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA Error at %s:%d — %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_err));          \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ================================================================
// Scalar copy: 4 bytes per thread per iteration
// ================================================================
__global__ void copyScalar(const float *src, float *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = src[i];
}

// ================================================================
// Vectorized copy: 16 bytes per thread per iteration (float4)
// Requires n to be divisible by 4 and base pointers 16-byte aligned.
// ================================================================
__global__ void copyFloat4(const float *src, float *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n / 4;

    if (i < n4) {
        // Reinterpret as float4 pointer — reads 16 bytes at once
        reinterpret_cast<float4*>(dst)[i] =
            reinterpret_cast<const float4*>(src)[i];
    }
}

// ================================================================
// __ldg() read-only cache: bypasses L1, uses texture/constant path
// Useful for read-only data with spatial locality or non-coalesced access
// ================================================================
__global__ void copyLdg(const float * __restrict__ src, float *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = __ldg(&src[i]);  // explicit read-only cache load
}

// ================================================================
// Vectorized + __ldg: float4 via __ldg
// ================================================================
__global__ void copyFloat4Ldg(const float * __restrict__ src, float *dst, int n)
{
    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n / 4;
    if (i < n4) {
        // Cast to float4 pointer and use __ldg for read-only cache
        float4 val = __ldg(reinterpret_cast<const float4*>(src) + i);
        reinterpret_cast<float4*>(dst)[i] = val;
    }
}

// ================================================================
// ILP-4 scalar copy: 4 independent streams per thread
// Hides memory latency by keeping 4 outstanding loads
// ================================================================
__global__ void copyILP4(const float *src, float *dst, int n)
{
    int stride = gridDim.x * blockDim.x;
    int base   = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes 4 elements, stepping by total grid size
    for (int i = base; i < n - 3 * stride; i += 4 * stride) {
        float v0 = src[i + 0 * stride];
        float v1 = src[i + 1 * stride];
        float v2 = src[i + 2 * stride];
        float v3 = src[i + 3 * stride];
        dst[i + 0 * stride] = v0;
        dst[i + 1 * stride] = v1;
        dst[i + 2 * stride] = v2;
        dst[i + 3 * stride] = v3;
    }
}

// ================================================================
// Timer
// ================================================================
float timeMs(void (*f)(cudaStream_t), int reps = 20)
{
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    f(0); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < reps; i++) f(0);
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaEventDestroy(s)); CUDA_CHECK(cudaEventDestroy(e));
    return ms / reps;
}

int main()
{
    // N must be divisible by 4 for float4 kernels
    const int N = 1 << 26;  // 64M floats = 256 MB
    size_t bytes = (size_t)N * sizeof(float);

    printf("Vectorized Memory Access Benchmark\n");
    printf("N = %d floats (%.0f MB)\n\n", N, bytes / 1e6);

    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));
    CUDA_CHECK(cudaMemset(d_src, 1, bytes));  // Non-zero so results differ

    int T = 256;  // Threads per block

    printf("%-30s  %10s  %10s\n", "Kernel", "Time (ms)", "BW (GB/s)");
    printf("%s\n", std::string(55, '-').c_str());

    auto report = [&](const char *name, float ms) {
        float bw = (2.0f * bytes) / (ms * 1e-3f) / 1e9f;
        printf("%-30s  %10.3f  %10.1f\n", name, ms, bw);
    };

    // Scalar
    int B1 = (N + T - 1) / T;
    report("Scalar (float)",
           timeMs([&](cudaStream_t s){ copyScalar<<<B1, T, 0, s>>>(d_src, d_dst, N); }));

    // float4 (N/4 threads, each loading 16 bytes)
    int B4 = (N / 4 + T - 1) / T;
    report("float4 (16B/thread)",
           timeMs([&](cudaStream_t s){ copyFloat4<<<B4, T, 0, s>>>(d_src, d_dst, N); }));

    // __ldg scalar
    report("Scalar + __ldg",
           timeMs([&](cudaStream_t s){ copyLdg<<<B1, T, 0, s>>>(d_src, d_dst, N); }));

    // float4 + __ldg
    report("float4 + __ldg",
           timeMs([&](cudaStream_t s){ copyFloat4Ldg<<<B4, T, 0, s>>>(d_src, d_dst, N); }));

    // ILP-4 with reduced grid (let each thread do 4 elements)
    int BILP = B1 / 4;
    report("Scalar ILP-4",
           timeMs([&](cudaStream_t s){ copyILP4<<<BILP, T, 0, s>>>(d_src, d_dst, N); }));

    printf("\nRTX 4090 theoretical peak bandwidth: ~1008 GB/s\n");
    printf("Note: cudaMemcpy achieves similar bandwidth to the best kernel above.\n");

    // Verify float4 == scalar
    float *h_scalar = (float*)malloc(bytes);
    float *h_float4 = (float*)malloc(bytes);
    CUDA_CHECK(cudaMemset(d_dst, 0, bytes));
    copyScalar<<<B1, T>>>(d_src, d_dst, N);
    CUDA_CHECK(cudaMemcpy(h_scalar, d_dst, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(d_dst, 0, bytes));
    copyFloat4<<<B4, T>>>(d_src, d_dst, N);
    CUDA_CHECK(cudaMemcpy(h_float4, d_dst, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    bool match = true;
    for (int i = 0; i < N; i++)
        if (h_scalar[i] != h_float4[i]) { match = false; break; }
    printf("\nScalar vs float4 results match: %s\n", match ? "YES" : "NO");

    CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_dst));
    free(h_scalar); free(h_float4);
    return 0;
}

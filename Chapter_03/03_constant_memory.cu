/*
 * Chapter 03 — 03_constant_memory.cu
 *
 * Demonstrates constant memory for a 1D convolution (FIR filter).
 * Filter coefficients are read-only and the same across all threads —
 * perfect for the constant memory broadcast mechanism.
 *
 * Compares:
 *   (a) Filter in global memory
 *   (b) Filter in constant memory
 *   (c) Filter in shared memory (loaded at kernel start)
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o const_mem 03_constant_memory.cu
 * Run:
 *   ./const_mem
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA Error at %s:%d — %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_err));          \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define FILTER_SIZE 17  // Must be odd; max ~64KB total for all constants

// ================================================================
// Constant memory filter — declared at file scope, not inside a function!
// The hardware has a dedicated constant cache (~8 KB per SM).
// When all threads in a warp read the same address, it's a single
// cached broadcast — effectively free.
// ================================================================
__constant__ float d_filter_const[FILTER_SIZE];

// ================================================================
// Version A: Filter in global memory (passed as pointer)
// ================================================================
__global__ void convGlobal(const float *in, float *out, int n,
                            const float *filter, int filterSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = filterSize / 2;
    if (i >= half && i < n - half) {
        float sum = 0.0f;
        for (int j = 0; j < filterSize; j++)
            sum += in[i - half + j] * filter[j];  // filter from slow global mem
        out[i] = sum;
    }
}

// ================================================================
// Version B: Filter in constant memory
// All threads read filter[j] simultaneously → broadcast, very fast
// ================================================================
__global__ void convConstant(const float *in, float *out, int n, int filterSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = filterSize / 2;
    if (i >= half && i < n - half) {
        float sum = 0.0f;
        for (int j = 0; j < filterSize; j++)
            sum += in[i - half + j] * d_filter_const[j];  // constant cache!
        out[i] = sum;
    }
}

// ================================================================
// Version C: Filter loaded into shared memory at start of each block
// Each block loads the filter once, then uses it from fast smem.
// ================================================================
__global__ void convSharedFilter(const float *in, float *out, int n,
                                 const float *filter, int filterSize)
{
    // Shared mem: filter + input tile (with halo)
    extern __shared__ float smem[];  // size = (filterSize + blockDim.x)*sizeof(float)
    float *sh_filter = smem;
    float *sh_input  = smem + filterSize;

    int half = filterSize / 2;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load filter into shared memory (first filterSize threads do it)
    if (threadIdx.x < filterSize)
        sh_filter[threadIdx.x] = filter[threadIdx.x];

    // Load input tile (including halo)
    sh_input[threadIdx.x + half] = (i < n) ? in[i] : 0.0f;
    if (threadIdx.x < half) {
        int gi = i - half;
        sh_input[threadIdx.x] = (gi >= 0) ? in[gi] : 0.0f;
    }
    if (threadIdx.x >= blockDim.x - half) {
        int gi = i + half;
        sh_input[threadIdx.x + 2 * half] = (gi < n) ? in[gi] : 0.0f;
    }

    __syncthreads();

    if (i >= half && i < n - half) {
        float sum = 0.0f;
        for (int j = 0; j < filterSize; j++)
            sum += sh_input[threadIdx.x + j] * sh_filter[j];
        out[i] = sum;
    }
}

// ================================================================
// Helper: time a kernel (returns average ms)
// ================================================================
float timeKernel(void (*f)(cudaStream_t), int reps = 20)
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

// ================================================================
// MAIN
// ================================================================
int main()
{
    const int N = 1 << 23;  // 8M elements
    size_t bytes = N * sizeof(float);

    printf("1D Convolution — Filter Memory Comparison\n");
    printf("N=%d, filter_size=%d\n\n", N, FILTER_SIZE);

    // --- Create Gaussian filter (sum = 1) ---
    float h_filter[FILTER_SIZE];
    float sum = 0.0f;
    int half = FILTER_SIZE / 2;
    for (int i = 0; i < FILTER_SIZE; i++) {
        float x = (float)(i - half);
        h_filter[i] = expf(-x * x / (2.0f * 4.0f * 4.0f));
        sum += h_filter[i];
    }
    for (int i = 0; i < FILTER_SIZE; i++) h_filter[i] /= sum;

    // --- Host data ---
    float *h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = sinf(0.01f * i);

    // --- Device data ---
    float *d_in, *d_out, *d_filter_global;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_filter_global, FILTER_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter_global, h_filter,
                          FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // --- Upload to constant memory ---
    // cudaMemcpyToSymbol: copies to a __constant__ variable by symbol name
    CUDA_CHECK(cudaMemcpyToSymbol(d_filter_const, h_filter,
                                  FILTER_SIZE * sizeof(float)));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    int smem_size = (FILTER_SIZE + threads + FILTER_SIZE) * sizeof(float);

    // --- A: Global memory filter ---
    auto global_launch = [&](cudaStream_t s) {
        convGlobal<<<blocks, threads, 0, s>>>(d_in, d_out, N,
                                              d_filter_global, FILTER_SIZE);
    };
    float ms_global = timeKernel(global_launch);

    // --- B: Constant memory filter ---
    auto const_launch = [&](cudaStream_t s) {
        convConstant<<<blocks, threads, 0, s>>>(d_in, d_out, N, FILTER_SIZE);
    };
    float ms_const = timeKernel(const_launch);

    // --- C: Shared memory filter ---
    auto shared_launch = [&](cudaStream_t s) {
        convSharedFilter<<<blocks, threads, smem_size, s>>>(
            d_in, d_out, N, d_filter_global, FILTER_SIZE);
    };
    float ms_shared = timeKernel(shared_launch);

    printf("%-30s  %8.3f ms\n", "Global memory filter:", ms_global);
    printf("%-30s  %8.3f ms   (%.2fx vs global)\n", "Constant memory filter:",
           ms_const, ms_global / ms_const);
    printf("%-30s  %8.3f ms   (%.2fx vs global)\n", "Shared memory filter:",
           ms_shared, ms_global / ms_shared);

    printf("\nNote: Constant memory wins when all threads read the same element.\n");
    printf("      Shared memory wins when each thread reads a different element.\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_filter_global));
    free(h_in);
    return 0;
}

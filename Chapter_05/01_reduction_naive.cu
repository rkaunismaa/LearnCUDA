/*
 * Chapter 05 — 01_reduction_naive.cu
 *
 * Parallel reduction: compute the sum of a large float array.
 * Compares interleaved addressing (divergent) vs sequential addressing (no divergence).
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o reduction_naive 01_reduction_naive.cu
 * Run:
 *   ./reduction_naive
 */

#include <stdio.h>
#include <stdlib.h>
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

const int BLOCK_SIZE = 256;

// ================================================================
// INTERLEAVED ADDRESSING (naive — divergent warps)
//
// stride:   1    2    4    8   ...
// active: 0,2  0,4  0,8  0,16  ...
//
// Threads 0 and 1 are in the same warp. Thread 1 is idle when
// thread 0 is active — divergence!
// ================================================================
__global__ void reduceInterleaved(float *g_data, int n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    // Load into shared memory
    sdata[tid] = (i < n) ? g_data[i] : 0.0f;
    __syncthreads();

    // Reduction with interleaved addressing — DIVERGENT
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0)       // Only every 2nd, 4th, 8th... thread active
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    // Write block result
    if (tid == 0) g_data[blockIdx.x] = sdata[0];
}

// ================================================================
// SEQUENTIAL ADDRESSING (no divergence)
//
// stride: blockDim/2   blockDim/4   ...   1
// active:   0..N/2-1   0..N/4-1   ...  0,1
//
// Active threads are always contiguous (no divergence)!
// ================================================================
__global__ void reduceSequential(float *g_data, int n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < n) ? g_data[i] : 0.0f;
    __syncthreads();

    // Reduction with sequential addressing — NO DIVERGENCE
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)                  // Lower half of threads always active
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) g_data[blockIdx.x] = sdata[0];
}

// ================================================================
// Host wrapper: run two-pass reduction (blocks → final sum)
// ================================================================
float reduceGPU(float *d_data, int n, bool sequential, float *timing_ms)
{
    // We'll modify d_data in-place, so copy it first
    float *d_work;
    CUDA_CHECK(cudaMalloc(&d_work, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_work, d_data, n * sizeof(float), cudaMemcpyDeviceToDevice));

    cudaEvent_t tS, tE;
    CUDA_CHECK(cudaEventCreate(&tS));
    CUDA_CHECK(cudaEventCreate(&tE));
    CUDA_CHECK(cudaEventRecord(tS));

    int remaining = n;
    while (remaining > 1) {
        int blocks = (remaining + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int smem   = BLOCK_SIZE * sizeof(float);
        if (sequential)
            reduceSequential<<<blocks, BLOCK_SIZE, smem>>>(d_work, remaining);
        else
            reduceInterleaved<<<blocks, BLOCK_SIZE, smem>>>(d_work, remaining);
        remaining = blocks;
    }

    CUDA_CHECK(cudaEventRecord(tE));
    CUDA_CHECK(cudaEventSynchronize(tE));
    CUDA_CHECK(cudaEventElapsedTime(timing_ms, tS, tE));
    CUDA_CHECK(cudaEventDestroy(tS));
    CUDA_CHECK(cudaEventDestroy(tE));

    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_work, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_work));
    return result;
}

int main()
{
    const int N = 1 << 24;  // 16M floats
    size_t bytes = N * sizeof(float);

    printf("Parallel Reduction: N = %d (%.1f MB)\n\n", N, bytes / 1e6f);

    // ---- Initialize host data ----
    float *h_data = (float*)malloc(bytes);
    srand(42);
    double cpu_sum = 0.0;
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(rand() % 100) / 100.0f;  // 0..1
        cpu_sum += h_data[i];
    }
    printf("CPU reference sum: %.4f\n\n", (float)cpu_sum);

    // ---- Copy to device ----
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // ---- Warmup ----
    float ms_tmp;
    reduceGPU(d_data, N, true, &ms_tmp);

    // ---- Run both versions ----
    float ms_interleaved, ms_sequential;

    float sum_interleaved = reduceGPU(d_data, N, false, &ms_interleaved);
    float sum_sequential  = reduceGPU(d_data, N, true,  &ms_sequential);

    printf("%-30s  %.3f ms  sum=%.4f  error=%.4f\n",
           "Interleaved (divergent):",
           ms_interleaved, sum_interleaved,
           fabsf(sum_interleaved - (float)cpu_sum));

    printf("%-30s  %.3f ms  sum=%.4f  error=%.4f\n",
           "Sequential (no divergence):",
           ms_sequential, sum_sequential,
           fabsf(sum_sequential - (float)cpu_sum));

    printf("\nSpeedup (sequential vs interleaved): %.2fx\n",
           ms_interleaved / ms_sequential);

    printf("\nNote: The error vs CPU sum is due to floating-point\n");
    printf("non-associativity (reduction order differs).\n");

    CUDA_CHECK(cudaFree(d_data));
    free(h_data);
    return 0;
}

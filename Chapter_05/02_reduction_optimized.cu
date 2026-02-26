/*
 * Chapter 05 — 02_reduction_optimized.cu
 *
 * Optimized parallel reduction using warp shuffle instructions.
 * No shared memory needed for the final warp-level reduction step.
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o reduction_opt 02_reduction_optimized.cu
 * Run:
 *   ./reduction_opt
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

// ================================================================
// WARP-LEVEL REDUCTION using shuffle (no shared memory)
// Each thread contributes its 'val'; result ends up in lane 0.
// ================================================================
__device__ __forceinline__ float warpReduceSum(float val)
{
    unsigned mask = 0xffffffff;
    // Each step: thread i gets sum of itself and thread i+offset
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);
    return val;
}

// ================================================================
// BLOCK-LEVEL REDUCTION
// Uses warp reduce + minimal shared memory (one float per warp).
// blockDim.x must be a multiple of 32.
// ================================================================
__device__ float blockReduceSum(float val)
{
    // Step 1: Reduce within each warp
    val = warpReduceSum(val);

    // Step 2: First lane of each warp writes its warp's sum to shared memory
    static __shared__ float warpSums[32];  // Max 1024/32 = 32 warps
    int lane   = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;

    if (lane == 0)
        warpSums[warpId] = val;
    __syncthreads();

    // Step 3: First warp reduces the per-warp sums
    int numWarps = blockDim.x / 32;
    val = (threadIdx.x < numWarps) ? warpSums[lane] : 0.0f;
    if (warpId == 0)
        val = warpReduceSum(val);

    return val;  // Only thread 0 has the correct block sum
}

// ================================================================
// OPTIMIZED REDUCTION KERNEL
// Each block reduces its portion to one partial sum.
// Load with unrolling: each thread loads multiple elements.
// ================================================================
__global__ void reduceOptimized(const float *g_in, float *g_out, int n)
{
    float sum = 0.0f;

    // Grid-stride loop: each thread sums multiple elements
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        sum += g_in[i];

    // Block-level reduction
    sum = blockReduceSum(sum);

    // Thread 0 of each block writes the block's partial sum
    if (threadIdx.x == 0)
        g_out[blockIdx.x] = sum;
}

// ================================================================
// HOST-SIDE TWO-PASS REDUCTION
// Pass 1: reduce full array to #blocks partial sums
// Pass 2: reduce partial sums to single value
// ================================================================
float reduceTwoPass(const float *d_in, int n, float *timing_ms)
{
    const int THREADS = 256;
    const int BLOCKS  = min(256, (n + THREADS - 1) / THREADS);

    float *d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, BLOCKS * sizeof(float)));

    cudaEvent_t tS, tE;
    CUDA_CHECK(cudaEventCreate(&tS));
    CUDA_CHECK(cudaEventCreate(&tE));
    CUDA_CHECK(cudaEventRecord(tS));

    // Pass 1: large array → BLOCKS partial sums
    reduceOptimized<<<BLOCKS, THREADS>>>(d_in, d_partial, n);

    // Pass 2: BLOCKS partial sums → 1 final sum
    reduceOptimized<<<1, THREADS>>>(d_partial, d_partial, BLOCKS);

    CUDA_CHECK(cudaEventRecord(tE));
    CUDA_CHECK(cudaEventSynchronize(tE));
    CUDA_CHECK(cudaEventElapsedTime(timing_ms, tS, tE));
    CUDA_CHECK(cudaEventDestroy(tS));
    CUDA_CHECK(cudaEventDestroy(tE));

    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_partial, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_partial));
    return result;
}

// ================================================================
// NAIVE SEQUENTIAL (from Ch05/01) for comparison
// ================================================================
__global__ void reduceSequential(float *g_data, int n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? g_data[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_data[blockIdx.x] = sdata[0];
}

float reduceNaive(float *d_data, int n, float *timing_ms)
{
    const int T = 256;
    float *d_work;
    CUDA_CHECK(cudaMalloc(&d_work, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_work, d_data, n * sizeof(float), cudaMemcpyDeviceToDevice));

    cudaEvent_t tS, tE;
    CUDA_CHECK(cudaEventCreate(&tS));
    CUDA_CHECK(cudaEventCreate(&tE));
    CUDA_CHECK(cudaEventRecord(tS));

    int rem = n;
    while (rem > 1) {
        int b = (rem + T - 1) / T;
        reduceSequential<<<b, T, T * sizeof(float)>>>(d_work, rem);
        rem = b;
    }

    CUDA_CHECK(cudaEventRecord(tE));
    CUDA_CHECK(cudaEventSynchronize(tE));
    CUDA_CHECK(cudaEventElapsedTime(timing_ms, tS, tE));
    CUDA_CHECK(cudaEventDestroy(tS));
    CUDA_CHECK(cudaEventDestroy(tE));

    float r;
    CUDA_CHECK(cudaMemcpy(&r, d_work, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_work));
    return r;
}

int main()
{
    const int N = 1 << 24;  // 16M floats
    size_t bytes = N * sizeof(float);

    printf("Optimized Reduction: N = %d (%.1f MB)\n\n", N, bytes / 1e6f);

    float *h_data = (float*)malloc(bytes);
    srand(42);
    double cpu_sum = 0.0;
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(rand() % 100) / 100.0f;
        cpu_sum += h_data[i];
    }
    printf("CPU reference sum: %.4f\n\n", (float)cpu_sum);

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // Warmup
    float ms_tmp; reduceTwoPass(d_data, N, &ms_tmp);

    float ms_naive, ms_opt;
    float sum_naive = reduceNaive(d_data, N, &ms_naive);
    float sum_opt   = reduceTwoPass(d_data, N, &ms_opt);

    printf("%-35s  %.3f ms  result=%.4f\n", "Naive (sequential, multi-pass):",
           ms_naive, sum_naive);
    printf("%-35s  %.3f ms  result=%.4f\n", "Optimized (warp shuffle, 2-pass):",
           ms_opt, sum_opt);
    printf("\nSpeedup:  %.2fx\n", ms_naive / ms_opt);
    printf("Error (optimized vs CPU): %.4f\n", fabsf(sum_opt - (float)cpu_sum));

    CUDA_CHECK(cudaFree(d_data));
    free(h_data);
    return 0;
}

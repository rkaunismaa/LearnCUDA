/*
 * Chapter 09 — 02_warp_shuffle.cu
 *
 * Demonstrates warp shuffle instructions:
 *   - __shfl_down_sync for warp-level reduction (sum, max)
 *   - __shfl_up_sync for prefix scan (inclusive sum)
 *   - __shfl_sync for broadcast from lane 0
 *   - __shfl_xor_sync for butterfly operations
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o warp_shuffle 02_warp_shuffle.cu
 * Run:
 *   ./warp_shuffle
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

#define FULL_MASK 0xffffffff

// ================================================================
// Warp-level sum using __shfl_down_sync
// No shared memory needed! No __syncthreads()!
// Result in lane 0 after all iterations.
// ================================================================
__device__ float warpReduceSum(float val)
{
    val += __shfl_down_sync(FULL_MASK, val, 16);
    val += __shfl_down_sync(FULL_MASK, val,  8);
    val += __shfl_down_sync(FULL_MASK, val,  4);
    val += __shfl_down_sync(FULL_MASK, val,  2);
    val += __shfl_down_sync(FULL_MASK, val,  1);
    return val;
}

// ================================================================
// Warp-level max using __shfl_down_sync
// ================================================================
__device__ float warpReduceMax(float val)
{
    val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, 16));
    val = fmaxf(val, __shfl_down_sync(FULL_MASK, val,  8));
    val = fmaxf(val, __shfl_down_sync(FULL_MASK, val,  4));
    val = fmaxf(val, __shfl_down_sync(FULL_MASK, val,  2));
    val = fmaxf(val, __shfl_down_sync(FULL_MASK, val,  1));
    return val;
}

// ================================================================
// Inclusive prefix scan within a warp using __shfl_up_sync
// Thread i gets the sum of all lanes 0..i
// ================================================================
__device__ float warpScan(float val)
{
    // Kogge-Stone parallel prefix scan
    for (int offset = 1; offset < 32; offset <<= 1) {
        float received = __shfl_up_sync(FULL_MASK, val, offset);
        int lane = threadIdx.x % 32;
        if (lane >= offset)
            val += received;
    }
    return val;  // Thread i now has inclusive prefix sum
}

// ================================================================
// Kernel to demonstrate reduction: each block → one partial sum
// ================================================================
__global__ void warpReduceKernel(const float *in, float *out, int n)
{
    float val = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        val += in[i];

    // Warp-level reduction
    val = warpReduceSum(val);

    // Per-warp results written to shared memory
    __shared__ float warp_results[32];
    int lane   = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    if (lane == 0) warp_results[warpId] = val;
    __syncthreads();

    // First warp reduces all warp results
    int nWarps = blockDim.x >> 5;
    val = (threadIdx.x < nWarps) ? warp_results[lane] : 0.0f;
    if (warpId == 0) val = warpReduceSum(val);

    if (threadIdx.x == 0) out[blockIdx.x] = val;
}

// ================================================================
// Kernel to demonstrate prefix scan on 32 elements (1 warp)
// ================================================================
__global__ void warpScanKernel(float *data, float *out)
{
    // Only 1 warp
    float val = data[threadIdx.x];
    float result = warpScan(val);
    out[threadIdx.x] = result;
}

// ================================================================
// Broadcast from lane 0 to all lanes
// ================================================================
__global__ void broadcastDemo(float *data, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Load value
    float val = data[i];

    // Lane 0's value is broadcast to all 32 lanes in the warp
    float lane0_val = __shfl_sync(FULL_MASK, val, 0);

    // Each thread uses lane 0's value as a scale factor
    out[i] = val / (lane0_val + 1e-8f);  // Normalize by first element
}

int main()
{
    // ================================================================
    // 1. Warp reduction: sum and max
    // ================================================================
    printf("=== Warp Shuffle Reduction ===\n");
    const int N = 1 << 24;  // 16M floats
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    int BLOCKS = 256, THREADS = 256;
    CUDA_CHECK(cudaMalloc(&d_out, BLOCKS * sizeof(float)));

    float *h_in = (float*)malloc(N * sizeof(float));
    srand(42);
    double cpu_sum = 0.0;
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)(rand() % 100) / 100.0f;
        cpu_sum += h_in[i];
    }
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // First pass: reduce to BLOCKS partial sums
    warpReduceKernel<<<BLOCKS, THREADS>>>(d_in, d_out, N);
    // Second pass: reduce BLOCKS partial sums to 1
    warpReduceKernel<<<1, THREADS>>>(d_out, d_out, BLOCKS);
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_sum;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    printf("CPU sum: %.4f\n", (float)cpu_sum);
    printf("GPU sum (warp shuffle): %.4f\n", gpu_sum);
    printf("Relative error: %.6f%%\n\n", fabsf(gpu_sum - (float)cpu_sum) / (float)cpu_sum * 100.0f);

    // ================================================================
    // 2. Warp scan on 32 elements
    // ================================================================
    printf("=== Warp Inclusive Prefix Scan (32 elements) ===\n");
    float h_scan_in[32], h_scan_out[32];
    for (int i = 0; i < 32; i++) h_scan_in[i] = (float)(i + 1);  // 1,2,3,...,32

    float *d_scan_in, *d_scan_out;
    CUDA_CHECK(cudaMalloc(&d_scan_in,  32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scan_out, 32 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_scan_in, h_scan_in, 32 * sizeof(float), cudaMemcpyHostToDevice));

    warpScanKernel<<<1, 32>>>(d_scan_in, d_scan_out);
    CUDA_CHECK(cudaMemcpy(h_scan_out, d_scan_out, 32 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Input:  [1, 2, 3, 4, ..., 32]\n");
    printf("Output: [");
    for (int i = 0; i < 8; i++) printf("%.0f%s", h_scan_out[i], i < 7 ? ", " : "...");
    printf("]\n");
    printf("Expected: [1, 3, 6, 10, 15, 21, 28, 36...]\n");
    printf("Last element (sum of 1..32): %.0f (expected 528)\n\n", h_scan_out[31]);

    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_scan_in)); CUDA_CHECK(cudaFree(d_scan_out));
    free(h_in);
    return 0;
}

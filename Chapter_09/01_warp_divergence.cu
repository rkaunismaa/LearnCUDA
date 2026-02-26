/*
 * Chapter 09 — 01_warp_divergence.cu
 *
 * Demonstrates warp divergence:
 *   - Divergent branch (threads in same warp take different paths)
 *   - Non-divergent equivalent
 *   - __ballot_sync to inspect which threads take each path
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o warp_div 01_warp_divergence.cu
 * Run:
 *   ./warp_div
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

const int N = 1 << 24;

// ================================================================
// DIVERGENT: thread's path depends on its index value
// Threads 0,2,4... take path A; threads 1,3,5... take path B
// Within a warp, both branches execute — 50% efficiency
// ================================================================
__global__ void divergentKernel(float *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (i % 2 == 0) {
            // Path A: polynomial evaluation
            float x = data[i];
            data[i] = x*x*x - 2.0f*x*x + 3.0f*x - 4.0f;
        } else {
            // Path B: different polynomial
            float x = data[i];
            data[i] = 4.0f*x*x + 2.0f*x + 1.0f;
        }
    }
}

// ================================================================
// NON-DIVERGENT: precompute branch, apply uniformly
// All threads in a warp do the same computation.
// Slight overhead from computing both results and selecting.
// ================================================================
__global__ void nonDivergentKernel(float *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = data[i];
        float res_a = x*x*x - 2.0f*x*x + 3.0f*x - 4.0f;
        float res_b = 4.0f*x*x + 2.0f*x + 1.0f;
        // Select based on condition — no branch, just a select
        data[i] = (i % 2 == 0) ? res_a : res_b;
    }
}

// ================================================================
// BALLOT_SYNC DEMO: inspect which threads take each path
// Only 1 block, 1 warp (32 threads) for clarity
// ================================================================
__global__ void ballotDemo(float *data, int n)
{
    // Only run with blockDim.x == 32, one warp
    int i = threadIdx.x;
    float x = data[i % n];

    // Condition: even-indexed threads
    bool condition = (i % 2 == 0);

    // Get ballot: which threads have condition=true?
    unsigned mask = 0xffffffff;
    unsigned ballot = __ballot_sync(mask, condition);

    // Thread 0 prints the ballot result
    if (i == 0) {
        printf("Ballot (even threads active): 0x%08X = ", ballot);
        // Print which lanes are set
        for (int lane = 0; lane < 32; lane++)
            printf("%d", (ballot >> lane) & 1);
        printf("\n");
        printf("Active lane count: %d / 32\n", __popc(ballot));
    }

    // __any_sync and __all_sync
    bool any_even  = __any_sync(mask, condition);
    bool all_even  = __all_sync(mask, condition);

    if (i == 0) {
        printf("__any_sync (any even?): %s\n", any_even ? "yes" : "no");
        printf("__all_sync (all even?): %s\n", all_even ? "yes" : "no");
    }

    data[i] = condition ? x * 2.0f : x + 1.0f;
}

float timeKernelMs(void (*f)(cudaStream_t), int reps = 10)
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
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(float)));

    int threads = 256, blocks = (N + 255) / 256;

    // ---- Ballot demo (1 warp) ----
    printf("=== Warp Ballot Demo (1 warp = 32 threads) ===\n");
    ballotDemo<<<1, 32>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Divergent vs non-divergent benchmark ----
    printf("\n=== Divergence Performance Impact ===\n");
    auto div_launch = [&](cudaStream_t s) {
        divergentKernel<<<blocks, threads, 0, s>>>(d_data, N);
    };
    auto nodiv_launch = [&](cudaStream_t s) {
        nonDivergentKernel<<<blocks, threads, 0, s>>>(d_data, N);
    };

    float ms_div   = timeKernelMs(div_launch);
    float ms_nodiv = timeKernelMs(nodiv_launch);

    printf("Divergent kernel:     %.3f ms\n", ms_div);
    printf("Non-divergent kernel: %.3f ms\n", ms_nodiv);
    printf("Speedup:              %.2fx\n\n", ms_div / ms_nodiv);

    printf("Note: Modern GPU microarchitectures partially mitigate divergence\n");
    printf("for simple cases. The gap is larger with more complex divergent code.\n");

    CUDA_CHECK(cudaFree(d_data));
    return 0;
}

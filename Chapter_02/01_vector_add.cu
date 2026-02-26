/*
 * Chapter 02 — 01_vector_add.cu
 *
 * Vector addition: C[i] = A[i] + B[i]
 * The canonical first CUDA program beyond Hello World.
 *
 * Demonstrates:
 *   - Full host/device memory workflow
 *   - 1D thread indexing
 *   - Timing with CUDA events
 *   - CPU vs GPU result verification
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o vec_add 01_vector_add.cu
 * Run:
 *   ./vec_add
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ---- Error checking macro (keep this in all your programs!) ----
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
// KERNEL: each thread adds one pair of elements
// ================================================================
__global__ void vecAddKernel(const float *A, const float *B, float *C, int n)
{
    // Compute this thread's global index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check — last block may have threads beyond array length
    if (i < n)
        C[i] = A[i] + B[i];
}

// ================================================================
// CPU reference implementation (for verification)
// ================================================================
void vecAddCPU(const float *A, const float *B, float *C, int n)
{
    for (int i = 0; i < n; i++)
        C[i] = A[i] + B[i];
}

// ================================================================
// MAIN
// ================================================================
int main()
{
    const int N = 1 << 24;  // 16M elements (~64 MB per float array)
    size_t bytes = N * sizeof(float);

    printf("Vector addition: N = %d (%.1f MB per array)\n\n",
           N, bytes / 1e6);

    // ---- Allocate host memory ----
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);      // GPU result (copied back)
    float *h_C_ref = (float*)malloc(bytes);  // CPU reference

    // Initialize input arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i / N;
        h_B[i] = (float)(N - i) / N;
    }

    // ---- Allocate device memory ----
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // ---- Copy input data to GPU ----
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // ---- Configure kernel launch ----
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    // Equivalent to: ceil(N / threadsPerBlock)

    printf("Launch config: %d blocks x %d threads = %d total threads\n\n",
           blocksPerGrid, threadsPerBlock, blocksPerGrid * threadsPerBlock);

    // ---- Time the kernel using CUDA events ----
    // (We'll cover events in depth in Chapter 07)
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    CUDA_CHECK(cudaEventRecord(startEvent));

    // ---- LAUNCH THE KERNEL ----
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));  // Wait for stop event

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));

    // ---- Copy result back to host ----
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // ---- CPU reference ----
    vecAddCPU(h_A, h_B, h_C_ref, N);

    // ---- Verify results ----
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_C[i] - h_C_ref[i]) > 1e-5f) {
            printf("ERROR at i=%d: GPU=%f, CPU=%f\n", i, h_C[i], h_C_ref[i]);
            errors++;
            if (errors > 5) break;
        }
    }
    if (errors == 0)
        printf("Results verified: GPU matches CPU for all %d elements.\n\n", N);

    // ---- Report performance ----
    printf("Kernel time:       %.3f ms\n", ms);
    // Memory bandwidth: we read 2 arrays and write 1 (3 * N * 4 bytes)
    double bw = (3.0 * bytes) / (ms * 1e-3) / 1e9;
    printf("Memory bandwidth:  %.1f GB/s\n", bw);
    // RTX 4090 peak bandwidth: ~1008 GB/s — this tells you how close to peak you are

    // ---- Cleanup ----
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C); free(h_C_ref);

    return 0;
}

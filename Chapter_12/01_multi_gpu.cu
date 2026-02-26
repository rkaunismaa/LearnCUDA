/*
 * Chapter 12 — 01_multi_gpu.cu
 *
 * Multi-GPU programming: split vector addition across available GPUs.
 * Gracefully handles systems with 1 or 2+ GPUs.
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o multi_gpu 01_multi_gpu.cu
 * Run:
 *   ./multi_gpu
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

__global__ void vecAddKernel(const float *A, const float *B, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

int main()
{
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    printf("Multi-GPU Vector Addition\n");
    printf("=========================\n");
    printf("Available GPUs: %d\n\n", deviceCount);

    for (int d = 0; d < deviceCount; d++) {
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, d));
        printf("  GPU %d: %s (CC %d.%d, %.1f GB)\n",
               d, p.name, p.major, p.minor, p.totalGlobalMem / 1e9f);
    }
    printf("\n");

    // ---- Check P2P capability ----
    if (deviceCount >= 2) {
        int canAccess01, canAccess10;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess01, 0, 1));
        CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess10, 1, 0));
        printf("P2P: GPU0→GPU1: %s  |  GPU1→GPU0: %s\n\n",
               canAccess01 ? "YES" : "NO",
               canAccess10 ? "YES" : "NO");
    }

    const int N = 1 << 26;  // 64M floats = 256 MB total per array
    size_t totalBytes = (size_t)N * sizeof(float);

    // ---- Host data ----
    float *h_A = (float*)malloc(totalBytes);
    float *h_B = (float*)malloc(totalBytes);
    float *h_C = (float*)malloc(totalBytes);  // Result from GPU(s)
    for (int i = 0; i < N; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    cudaEvent_t tStart, tStop;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaEventCreate(&tStart));
    CUDA_CHECK(cudaEventCreate(&tStop));

    // ================================================================
    // A: SINGLE GPU (GPU 0)
    // ================================================================
    {
        CUDA_CHECK(cudaSetDevice(0));
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, totalBytes));
        CUDA_CHECK(cudaMalloc(&d_B, totalBytes));
        CUDA_CHECK(cudaMalloc(&d_C, totalBytes));

        CUDA_CHECK(cudaEventRecord(tStart));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, totalBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, totalBytes, cudaMemcpyHostToDevice));
        vecAddKernel<<<(N + 255)/256, 256>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaMemcpy(h_C, d_C, totalBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(tStop));
        CUDA_CHECK(cudaEventSynchronize(tStop));

        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, tStart, tStop));
        printf("Single GPU (GPU 0): %.3f ms  |  result[0]=%.1f\n", ms, h_C[0]);

        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    }

    // ================================================================
    // B: MULTI-GPU (if 2+ GPUs available)
    // ================================================================
    if (deviceCount >= 2) {
        int numGPUs = min(deviceCount, 2);
        int chunkSize = N / numGPUs;  // Elements per GPU

        float  *d_A[2], *d_B[2], *d_C[2];
        cudaStream_t streams[2];

        // Allocate on each GPU
        for (int g = 0; g < numGPUs; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            size_t chunkBytes = (size_t)chunkSize * sizeof(float);
            CUDA_CHECK(cudaMalloc(&d_A[g], chunkBytes));
            CUDA_CHECK(cudaMalloc(&d_B[g], chunkBytes));
            CUDA_CHECK(cudaMalloc(&d_C[g], chunkBytes));
            CUDA_CHECK(cudaStreamCreate(&streams[g]));
        }

        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaEventRecord(tStart));

        // Launch on each GPU
        for (int g = 0; g < numGPUs; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            size_t offset      = (size_t)g * chunkSize;
            size_t chunkBytes  = (size_t)chunkSize * sizeof(float);
            int    blocks      = (chunkSize + 255) / 256;

            CUDA_CHECK(cudaMemcpyAsync(d_A[g], h_A + offset, chunkBytes,
                                       cudaMemcpyHostToDevice, streams[g]));
            CUDA_CHECK(cudaMemcpyAsync(d_B[g], h_B + offset, chunkBytes,
                                       cudaMemcpyHostToDevice, streams[g]));
            vecAddKernel<<<blocks, 256, 0, streams[g]>>>(
                d_A[g], d_B[g], d_C[g], chunkSize);
            CUDA_CHECK(cudaMemcpyAsync(h_C + offset, d_C[g], chunkBytes,
                                       cudaMemcpyDeviceToHost, streams[g]));
        }

        // Synchronize all GPUs
        for (int g = 0; g < numGPUs; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaStreamSynchronize(streams[g]));
        }

        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaEventRecord(tStop));
        CUDA_CHECK(cudaEventSynchronize(tStop));

        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, tStart, tStop));
        printf("Multi-GPU (%d GPUs):  %.3f ms  |  result[0]=%.1f  result[N-1]=%.1f\n",
               numGPUs, ms, h_C[0], h_C[N-1]);

        // Verify
        bool ok = true;
        for (int i = 0; i < N; i++)
            if (fabsf(h_C[i] - 3.0f) > 1e-4f) { ok = false; break; }
        printf("Verification: %s\n", ok ? "PASSED" : "FAILED");

        for (int g = 0; g < numGPUs; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaFree(d_A[g])); CUDA_CHECK(cudaFree(d_B[g]));
            CUDA_CHECK(cudaFree(d_C[g])); CUDA_CHECK(cudaStreamDestroy(streams[g]));
        }
    } else {
        printf("\nOnly 1 GPU available — multi-GPU test skipped.\n");
    }

    CUDA_CHECK(cudaEventDestroy(tStart));
    CUDA_CHECK(cudaEventDestroy(tStop));
    free(h_A); free(h_B); free(h_C);
    return 0;
}

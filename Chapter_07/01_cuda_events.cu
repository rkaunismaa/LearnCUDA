/*
 * Chapter 07 — 01_cuda_events.cu
 *
 * Demonstrates CUDA events for precise kernel timing.
 * Shows timing of: kernel-only, H2D+kernel+D2H, and bandwidth calculations.
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o cuda_events 01_cuda_events.cu
 * Run:
 *   ./cuda_events
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
// A simple memory-bandwidth kernel (copy)
// ================================================================
__global__ void copyKernel(const float *src, float *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// ================================================================
// A compute-intensive kernel (FMA loop)
// ================================================================
__global__ void computeKernel(float *data, int n, int iters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = data[i];
        for (int k = 0; k < iters; k++)
            v = v * 1.0001f + 0.00001f;  // fused multiply-add (FMA)
        data[i] = v;
    }
}

// ================================================================
// Generic timing wrapper using CUDA events
// Returns average time in milliseconds over `reps` runs
// ================================================================
template<typename F>
float timeMs(F kernel_launch, int reps = 10, cudaStream_t stream = 0)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup run (cold cache / JIT compilation effects)
    kernel_launch(stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int r = 0; r < reps; r++)
        kernel_launch(stream);
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));  // Block CPU until GPU is done

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / reps;
}

// ================================================================
// Measure full round-trip: H2D + kernel + D2H
// ================================================================
float timeRoundTrip(const float *h_src, float *h_dst,
                    float *d_src, float *d_dst, int n,
                    int threads, int blocks)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    size_t bytes = n * sizeof(float);

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));
    copyKernel<<<blocks, threads>>>(d_src, d_dst, n);
    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

int main()
{
    const int N = 1 << 25;  // 32M floats = 128 MB
    size_t bytes = N * sizeof(float);
    int threads = 256, blocks = (N + 255) / 256;

    printf("CUDA Event Timing Demonstration\n");
    printf("N = %d floats (%.0f MB)\n\n", N, bytes / 1e6);

    // ---- Allocate ----
    float *h_src = (float*)malloc(bytes);
    float *h_dst = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_src[i] = (float)i / N;

    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));

    // ================================================================
    // 1. Kernel-only timing (data already on GPU)
    // ================================================================
    printf("--- Kernel-Only Timing ---\n");
    {
        auto launch = [&](cudaStream_t s) {
            copyKernel<<<blocks, threads, 0, s>>>(d_src, d_dst, N);
        };
        float ms = timeMs(launch, 20);
        float bw = (2.0 * bytes) / (ms * 1e-3) / 1e9;  // read + write
        printf("Copy kernel:   %.3f ms   bandwidth: %.1f GB/s\n", ms, bw);
        printf("(RTX 4090 peak: ~1008 GB/s)\n\n");
    }

    // ================================================================
    // 2. Compute kernel at different iteration counts
    // ================================================================
    printf("--- Compute Kernel Scaling ---\n");
    CUDA_CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));
    for (int iters : {10, 100, 1000}) {
        auto launch = [&](cudaStream_t s) {
            computeKernel<<<blocks, threads, 0, s>>>(d_src, N, iters);
        };
        float ms = timeMs(launch, 5);
        // FLOPs: N * iters * 2 (one FMA = 2 ops)
        double gflops = (2.0 * N * iters) / (ms * 1e-3) / 1e9;
        printf("iters=%4d:  %.3f ms   %.1f GFLOPS\n", iters, ms, gflops);
    }
    printf("(RTX 4090 FP32 peak: ~82,600 GFLOPS)\n\n");

    // ================================================================
    // 3. Full round-trip timing (includes H2D + D2H transfer overhead)
    // ================================================================
    printf("--- Full Round-Trip (H2D + Kernel + D2H) ---\n");
    {
        float ms = timeRoundTrip(h_src, h_dst, d_src, d_dst, N, threads, blocks);
        float kernel_ms_only = timeMs([&](cudaStream_t s){
            copyKernel<<<blocks,threads,0,s>>>(d_src, d_dst, N);
        }, 1);
        printf("Round-trip time: %.3f ms\n", ms);
        printf("Kernel-only:     %.3f ms\n", kernel_ms_only);
        printf("Transfer overhead: %.3f ms (%.1f%% of total)\n",
               ms - kernel_ms_only, (ms - kernel_ms_only) / ms * 100);
    }

    // ================================================================
    // 4. Events across streams (measuring concurrent operations)
    // ================================================================
    printf("\n--- Events Across Streams ---\n");
    {
        cudaStream_t s1, s2;
        CUDA_CHECK(cudaStreamCreate(&s1));
        CUDA_CHECK(cudaStreamCreate(&s2));

        float *d_a, *d_b;
        CUDA_CHECK(cudaMalloc(&d_a, bytes / 2));
        CUDA_CHECK(cudaMalloc(&d_b, bytes / 2));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, s1));
        // Two kernels on different streams — may run concurrently
        copyKernel<<<blocks/2, threads, 0, s1>>>(d_src, d_a, N/2);
        copyKernel<<<blocks/2, threads, 0, s2>>>(d_src + N/2, d_b, N/2);
        cudaStreamSynchronize(s1);
        cudaStreamSynchronize(s2);
        CUDA_CHECK(cudaEventRecord(stop, s1));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("Two concurrent half-array copies: %.3f ms\n", ms);
        printf("(vs single full-array kernel: run copy_kernel to compare)\n");

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaStreamDestroy(s1));
        CUDA_CHECK(cudaStreamDestroy(s2));
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
    }

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    free(h_src); free(h_dst);
    return 0;
}

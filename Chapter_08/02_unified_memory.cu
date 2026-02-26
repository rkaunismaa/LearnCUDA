/*
 * Chapter 08 — 02_unified_memory.cu
 *
 * Demonstrates Unified Memory (cudaMallocManaged):
 *   - No explicit cudaMemcpy needed
 *   - Page fault behavior (slow without prefetch)
 *   - cudaMemPrefetchAsync (fast, avoids page faults)
 *   - cudaMemAdvise usage
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o unified_memory 02_unified_memory.cu
 * Run:
 *   ./unified_memory
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

// A simple kernel that works on any pointer — UM or explicit device ptr
__global__ void vectorAdd(float *a, float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void initKernel(float *data, float val, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = val + (float)i / n;
}

int main()
{
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    const int N = 1 << 22;  // 4M floats = 16 MB per array
    size_t bytes = N * sizeof(float);
    int threads = 256, blocks = (N + 255) / 256;

    printf("Unified Memory Demo: N = %d floats (%.0f MB per array)\n\n", N, bytes/1e6);

    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));

    // ================================================================
    // METHOD A: Traditional explicit copies (baseline)
    // ================================================================
    printf("--- A: Traditional explicit cudaMemcpy ---\n");
    {
        float *h_a = (float*)malloc(bytes);
        float *h_b = (float*)malloc(bytes);
        float *h_c = (float*)malloc(bytes);
        for (int i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

        float *d_a, *d_b, *d_c;
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));

        CUDA_CHECK(cudaEventRecord(s));
        CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
        vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));

        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        printf("  Total time (H2D + kernel + D2H): %.3f ms\n", ms);
        printf("  Result check: c[0]=%.1f c[N-1]=%.1f\n\n", h_c[0], h_c[N-1]);

        CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_c));
        free(h_a); free(h_b); free(h_c);
    }

    // ================================================================
    // METHOD B: Unified Memory WITHOUT prefetch (page faults occur)
    // ================================================================
    printf("--- B: Unified Memory (no prefetch — page faults) ---\n");
    {
        float *a, *b, *c;
        CUDA_CHECK(cudaMallocManaged(&a, bytes));
        CUDA_CHECK(cudaMallocManaged(&b, bytes));
        CUDA_CHECK(cudaMallocManaged(&c, bytes));

        // CPU writes initial values — pages start on CPU
        for (int i = 0; i < N; i++) { a[i] = 1.0f; b[i] = 2.0f; }

        CUDA_CHECK(cudaEventRecord(s));
        // GPU accesses pages not yet migrated → page faults → slow!
        vectorAdd<<<blocks, threads>>>(a, b, c, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));

        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        printf("  Kernel time (includes page faults): %.3f ms\n", ms);
        printf("  Result check: c[0]=%.1f c[N-1]=%.1f\n\n", c[0], c[N-1]);

        CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b)); CUDA_CHECK(cudaFree(c));
    }

    // ================================================================
    // METHOD C: Unified Memory WITH prefetch (no page faults)
    // ================================================================
    printf("--- C: Unified Memory + cudaMemPrefetchAsync (no page faults) ---\n");
    {
        float *a, *b, *c;
        CUDA_CHECK(cudaMallocManaged(&a, bytes));
        CUDA_CHECK(cudaMallocManaged(&b, bytes));
        CUDA_CHECK(cudaMallocManaged(&c, bytes));

        // CPU initializes — pages on CPU
        for (int i = 0; i < N; i++) { a[i] = 1.0f; b[i] = 2.0f; }

        // Prefetch to GPU BEFORE kernel — no page faults during kernel
        CUDA_CHECK(cudaMemPrefetchAsync(a, bytes, device));
        CUDA_CHECK(cudaMemPrefetchAsync(b, bytes, device));
        CUDA_CHECK(cudaMemPrefetchAsync(c, bytes, device));

        CUDA_CHECK(cudaEventRecord(s));
        vectorAdd<<<blocks, threads>>>(a, b, c, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));

        float ms_kernel; CUDA_CHECK(cudaEventElapsedTime(&ms_kernel, s, e));
        printf("  Kernel time (no page faults): %.3f ms\n", ms_kernel);

        // Prefetch back to CPU before reading
        CUDA_CHECK(cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId));
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("  Result check: c[0]=%.1f c[N-1]=%.1f\n\n", c[0], c[N-1]);

        CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b)); CUDA_CHECK(cudaFree(c));
    }

    // ================================================================
    // METHOD D: Init on GPU, read on CPU — demonstrates bidirectional
    // ================================================================
    printf("--- D: GPU init → CPU read (bidirectional UM) ---\n");
    {
        float *data;
        CUDA_CHECK(cudaMallocManaged(&data, bytes));

        // GPU initializes the data
        initKernel<<<blocks, threads>>>(data, 100.0f, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Prefetch back to CPU for reading
        CUDA_CHECK(cudaMemPrefetchAsync(data, bytes, cudaCpuDeviceId));
        CUDA_CHECK(cudaDeviceSynchronize());

        // CPU reads without any memcpy!
        printf("  data[0]=%.4f  data[N/2]=%.4f  data[N-1]=%.4f\n",
               data[0], data[N/2], data[N-1]);
        printf("  (GPU initialized, CPU read — no memcpy needed)\n\n");

        CUDA_CHECK(cudaFree(data));
    }

    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    printf("Summary:\n");
    printf("  - Explicit cudaMemcpy: most control, best for predictable access\n");
    printf("  - UM without prefetch: simplest code, slowest (page faults)\n");
    printf("  - UM with prefetch:    simple code, near-explicit-copy performance\n");
    return 0;
}

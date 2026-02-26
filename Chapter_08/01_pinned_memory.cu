/*
 * Chapter 08 — 01_pinned_memory.cu
 *
 * Benchmarks pageable vs pinned (page-locked) host memory for
 * H2D and D2H transfers at various data sizes.
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o pinned_memory 01_pinned_memory.cu
 * Run:
 *   ./pinned_memory
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
// Benchmark transfer bandwidth between host and device
// ================================================================
float measureBandwidth(void *h_mem, void *d_mem, size_t bytes,
                       cudaMemcpyKind kind, bool async, int reps = 10)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    if (async) {
        cudaMemcpyAsync(d_mem, h_mem, bytes, kind);
        cudaDeviceSynchronize();
    } else {
        cudaMemcpy(d_mem, h_mem, bytes, kind);
    }

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < reps; r++) {
        if (async)
            cudaMemcpyAsync(d_mem, h_mem, bytes, kind);
        else
            cudaMemcpy(d_mem, h_mem, bytes, kind);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= reps;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return (float)bytes / (ms * 1e-3f) / 1e9f;  // GB/s
}

int main()
{
    printf("Pinned vs Pageable Memory Transfer Benchmark\n\n");
    printf("%-10s  %-20s  %-20s  %-15s\n", "Size", "Pageable H2D", "Pinned H2D", "Speedup");
    printf("%s\n", std::string(70, '-').c_str());

    // Test sizes from 1 MB to 1 GB
    size_t sizes[] = {1<<20, 4<<20, 16<<20, 64<<20, 256<<20, 512<<20};
    const char *size_names[] = {"1 MB", "4 MB", "16 MB", "64 MB", "256 MB", "512 MB"};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    float *d_data;
    // Allocate max device buffer
    CUDA_CHECK(cudaMalloc(&d_data, sizes[nsizes - 1]));

    for (int s = 0; s < nsizes; s++) {
        size_t bytes = sizes[s];

        // Pageable host memory (regular malloc)
        void *h_pageable = malloc(bytes);

        // Pinned host memory (page-locked)
        void *h_pinned;
        CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));

        // Initialize both to same values
        memset(h_pageable, 1, bytes);
        memset(h_pinned, 1, bytes);

        // Measure H2D bandwidth
        float bw_pageable_h2d = measureBandwidth(h_pageable, d_data, bytes,
                                                  cudaMemcpyHostToDevice, false);
        float bw_pinned_h2d   = measureBandwidth(h_pinned, d_data, bytes,
                                                  cudaMemcpyHostToDevice, false);

        printf("%-10s  %-20.1f  %-20.1f  %.2fx\n",
               size_names[s], bw_pageable_h2d, bw_pinned_h2d,
               bw_pinned_h2d / bw_pageable_h2d);

        free(h_pageable);
        CUDA_CHECK(cudaFreeHost(h_pinned));
    }

    printf("\n--- D2H Bandwidth ---\n");
    printf("%-10s  %-20s  %-20s  %-15s\n", "Size", "Pageable D2H", "Pinned D2H", "Speedup");
    printf("%s\n", std::string(70, '-').c_str());

    for (int s = 0; s < nsizes; s++) {
        size_t bytes = sizes[s];

        void *h_pageable = malloc(bytes);
        void *h_pinned;
        CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));

        float bw_pageable_d2h = measureBandwidth(d_data, h_pageable, bytes,
                                                  cudaMemcpyDeviceToHost, false);
        float bw_pinned_d2h   = measureBandwidth(d_data, h_pinned, bytes,
                                                  cudaMemcpyDeviceToHost, false);

        printf("%-10s  %-20.1f  %-20.1f  %.2fx\n",
               size_names[s], bw_pageable_d2h, bw_pinned_d2h,
               bw_pinned_d2h / bw_pageable_d2h);

        free(h_pageable);
        CUDA_CHECK(cudaFreeHost(h_pinned));
    }

    // ---- Demonstrate async requirement: pinned only ----
    printf("\n--- Async Transfer (requires pinned memory) ---\n");
    {
        size_t bytes = 64 << 20;  // 64 MB
        void *h_pinned;
        CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));
        memset(h_pinned, 0, bytes);

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        cudaEvent_t s, e;
        CUDA_CHECK(cudaEventCreate(&s));
        CUDA_CHECK(cudaEventCreate(&e));

        // With pinned memory, this is truly non-blocking on the host
        CUDA_CHECK(cudaEventRecord(s, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_data, h_pinned, bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaEventRecord(e, stream));
        CUDA_CHECK(cudaEventSynchronize(e));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        printf("Async H2D (64 MB, pinned): %.3f ms  =  %.1f GB/s\n",
               ms, bytes / (ms * 1e-3f) / 1e9f);

        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaEventDestroy(s));
        CUDA_CHECK(cudaEventDestroy(e));
        CUDA_CHECK(cudaFreeHost(h_pinned));
    }

    printf("\nNote: PCIe 4.0 x16 theoretical max: ~32 GB/s bidirectional\n");
    CUDA_CHECK(cudaFree(d_data));
    return 0;
}

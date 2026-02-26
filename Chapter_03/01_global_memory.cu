/*
 * Chapter 03 — 01_global_memory.cu
 *
 * Demonstrates the impact of memory access patterns on performance.
 * Compares coalesced vs strided vs random global memory access.
 *
 * Key insight: for memory-bound kernels, HOW you access memory
 * matters far more than the arithmetic you perform.
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o global_memory 01_global_memory.cu
 * Run:
 *   ./global_memory
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

// Number of elements (must be large enough to see bandwidth differences)
const int N = 1 << 25;  // 32M floats = 128 MB

// ================================================================
// COALESCED: Thread i reads element i (consecutive addresses in warp)
// Best case: hardware combines 32 consecutive 4-byte reads into
// a single 128-byte cache line transaction.
// ================================================================
__global__ void copyCoalesced(float *__restrict__ dst,
                               const float *__restrict__ src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = src[i];
}

// ================================================================
// STRIDED: Thread i reads element i*stride (non-consecutive)
// Wastes memory bandwidth — hardware still fetches full cache lines
// but only one element per line is used.
// Stride=2: 50% utilization. Stride=32: 3% utilization.
// ================================================================
__global__ void copyStrided(float *__restrict__ dst,
                             const float *__restrict__ src, int n, int stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * stride;
    if (idx < n)
        dst[i] = src[idx];
}

// ================================================================
// OFFSET: All threads shifted by a fixed offset.
// With 4-byte floats, any offset that's a multiple of 32 is still
// coalesced (aligns to 128-byte cache line boundary).
// An offset of 1 misaligns by 4 bytes — degrades performance.
// ================================================================
__global__ void copyOffset(float *__restrict__ dst,
                            const float *__restrict__ src, int n, int offset)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - offset)
        dst[i] = src[i + offset];
}

// ================================================================
// Helper: time a kernel and report bandwidth
// ================================================================
float timeBandwidth(const char *label,
                    void (*launcher)(cudaStream_t),
                    long long bytes_transferred,
                    int repeats = 5)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    launcher(0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < repeats; r++)
        launcher(0);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= repeats;

    float bw = (float)bytes_transferred / (ms * 1e-3f) / 1e9f;
    printf("  %-35s  %7.3f ms   %7.1f GB/s\n", label, ms, bw);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return bw;
}

// ================================================================
// MAIN
// ================================================================
int main()
{
    printf("Global Memory Access Patterns\n");
    printf("N = %d floats (%.0f MB per array)\n\n", N, N * 4.0 / 1e6);

    size_t bytes = (long long)N * sizeof(float);

    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));

    // Initialize source
    CUDA_CHECK(cudaMemset(d_src, 0, bytes));
    CUDA_CHECK(cudaMemset(d_dst, 0, bytes));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    printf("%-35s  %10s   %10s\n", "Pattern", "Time", "Bandwidth");
    printf("%s\n", std::string(60, '-').c_str());

    // --- Coalesced (baseline) ---
    auto coalesced_launch = [&](cudaStream_t s) {
        copyCoalesced<<<blocks, threads, 0, s>>>(d_dst, d_src, N);
    };
    float peak = timeBandwidth("Coalesced (stride=1)", coalesced_launch,
                               2LL * bytes);

    // --- Strided access patterns ---
    for (int stride : {2, 4, 8, 32}) {
        int sblocks = (N / stride + threads - 1) / threads;
        char label[64];
        snprintf(label, sizeof(label), "Strided (stride=%d)", stride);
        auto launch = [&](cudaStream_t s) {
            copyStrided<<<sblocks, threads, 0, s>>>(d_dst, d_src, N, stride);
        };
        // Bytes actually transferred / useful: different counts
        timeBandwidth(label, launch, 2LL * (N / stride) * sizeof(float));
    }

    // --- Offset access ---
    printf("\nOffset effects (coalesced but misaligned):\n");
    for (int offset : {0, 1, 8, 16, 32}) {
        char label[64];
        snprintf(label, sizeof(label), "Offset=%d", offset);
        auto launch = [&](cudaStream_t s) {
            copyOffset<<<blocks, threads, 0, s>>>(d_dst, d_src, N, offset);
        };
        timeBandwidth(label, launch, 2LL * bytes);
    }

    printf("\nPeak bandwidth achieved: %.1f GB/s\n", peak);
    printf("RTX 4090 theoretical peak: ~1008 GB/s\n");

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    return 0;
}

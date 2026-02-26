/*
 * Chapter 05 — 03_atomics.cu
 *
 * Demonstrates atomic operations:
 *   - atomicAdd for parallel histogram
 *   - Privatized histogram (shared memory) to reduce contention
 *   - atomicMax for finding the maximum
 *   - atomicCAS to implement custom atomicMinFloat
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o atomics 03_atomics.cu
 * Run:
 *   ./atomics
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

#define NUM_BINS 256

// ================================================================
// NAIVE HISTOGRAM: direct global atomic (high contention)
// Many threads compete for the same bin → serialized
// ================================================================
__global__ void histogramNaive(const unsigned char *data, int *hist, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(&hist[data[i]], 1);  // All threads fight over 256 bins
}

// ================================================================
// PRIVATIZED HISTOGRAM: shared memory per block, merge at end
// Each block accumulates into its own private copy in shared memory
// — much less contention since 256 threads typically spread across bins
// ================================================================
__global__ void histogramPrivatized(const unsigned char *data, int *hist, int n)
{
    // Private per-block histogram in fast shared memory
    __shared__ int smem_hist[NUM_BINS];

    // Initialize shared histogram to 0
    if (threadIdx.x < NUM_BINS)
        smem_hist[threadIdx.x] = 0;
    __syncthreads();

    // Accumulate in shared memory (low contention — only block-local threads)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(&smem_hist[data[i]], 1);
    __syncthreads();

    // Merge block's result into global histogram
    if (threadIdx.x < NUM_BINS)
        atomicAdd(&hist[threadIdx.x], smem_hist[threadIdx.x]);
}

// ================================================================
// ATOMIC MAX for integers
// Find the maximum value in an array using atomicMax
// ================================================================
__global__ void findMax(const int *data, int *max_val, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicMax(max_val, data[i]);
}

// ================================================================
// atomicCAS-based float minimum
// atomicMin for float doesn't exist natively in CUDA.
// We implement it using atomicCAS (compare-and-swap):
//   - Read current value
//   - If our value is smaller, try to swap it in
//   - Retry if another thread changed the value between our read and CAS
// ================================================================
__device__ float atomicMinFloat(float *addr, float val)
{
    // Reinterpret float bits as int for atomicCAS
    int *addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float current = __int_as_float(expected);
        if (val >= current) break;  // Our value isn't smaller, stop
        int new_val = __float_as_int(val);
        old = atomicCAS(addr_as_int, expected, new_val);
    } while (old != expected);  // Retry if CAS failed (another thread changed it)
    return __int_as_float(old);
}

__global__ void findMinFloat(const float *data, float *min_val, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicMinFloat(min_val, data[i]);
}

// ================================================================
// Timing helper
// ================================================================
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
    // ================================================================
    // Part 1: Histogram (naive vs privatized)
    // ================================================================
    printf("=== Parallel Histogram ===\n");
    const int N_HIST = 1 << 22;  // 4M bytes
    unsigned char *h_bytes = (unsigned char*)malloc(N_HIST);
    srand(42);
    for (int i = 0; i < N_HIST; i++) h_bytes[i] = rand() % NUM_BINS;

    unsigned char *d_bytes;
    int *d_hist;
    CUDA_CHECK(cudaMalloc(&d_bytes, N_HIST));
    CUDA_CHECK(cudaMalloc(&d_hist, NUM_BINS * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_bytes, h_bytes, N_HIST, cudaMemcpyHostToDevice));

    int threads = 256, blocks = (N_HIST + 255) / 256;

    // Naive histogram timing
    auto naive_launch = [&](cudaStream_t s) {
        CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(int)));
        histogramNaive<<<blocks, threads, 0, s>>>(d_bytes, d_hist, N_HIST);
    };
    float ms_naive = timeKernelMs(naive_launch);

    // Verify naive
    int *h_hist_naive = (int*)malloc(NUM_BINS * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_hist_naive, d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost));

    // Privatized histogram timing
    auto priv_launch = [&](cudaStream_t s) {
        CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(int)));
        histogramPrivatized<<<blocks, threads, 0, s>>>(d_bytes, d_hist, N_HIST);
    };
    float ms_priv = timeKernelMs(priv_launch);

    int *h_hist_priv = (int*)malloc(NUM_BINS * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_hist_priv, d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify both match
    bool hist_match = true;
    for (int b = 0; b < NUM_BINS; b++)
        if (h_hist_naive[b] != h_hist_priv[b]) { hist_match = false; break; }

    printf("Naive histogram:       %.3f ms\n", ms_naive);
    printf("Privatized histogram:  %.3f ms  (%.2fx speedup)\n",
           ms_priv, ms_naive / ms_priv);
    printf("Results match: %s\n\n", hist_match ? "YES" : "NO");

    // ================================================================
    // Part 2: atomicMax for integers
    // ================================================================
    printf("=== atomicMax (integer) ===\n");
    const int N_MAX = 1 << 20;  // 1M integers
    int *h_ints = (int*)malloc(N_MAX * sizeof(int));
    int true_max = INT_MIN;
    for (int i = 0; i < N_MAX; i++) {
        h_ints[i] = rand() % 1000000;
        if (h_ints[i] > true_max) true_max = h_ints[i];
    }

    int *d_ints, *d_max;
    CUDA_CHECK(cudaMalloc(&d_ints, N_MAX * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_ints, h_ints, N_MAX * sizeof(int), cudaMemcpyHostToDevice));

    int init = INT_MIN;
    CUDA_CHECK(cudaMemcpy(d_max, &init, sizeof(int), cudaMemcpyHostToDevice));
    findMax<<<(N_MAX + 255) / 256, 256>>>(d_ints, d_max, N_MAX);
    CUDA_CHECK(cudaDeviceSynchronize());

    int gpu_max;
    CUDA_CHECK(cudaMemcpy(&gpu_max, d_max, sizeof(int), cudaMemcpyDeviceToHost));
    printf("CPU max: %d  |  GPU max: %d  |  Match: %s\n\n",
           true_max, gpu_max, (true_max == gpu_max) ? "YES" : "NO");

    // ================================================================
    // Part 3: atomicCAS-based float minimum
    // ================================================================
    printf("=== atomicMinFloat (via atomicCAS) ===\n");
    const int N_FLOAT = 1 << 20;
    float *h_floats = (float*)malloc(N_FLOAT * sizeof(float));
    float true_min = 1e30f;
    for (int i = 0; i < N_FLOAT; i++) {
        h_floats[i] = (float)rand() / RAND_MAX * 1000.0f;
        if (h_floats[i] < true_min) true_min = h_floats[i];
    }

    float *d_floats, *d_min;
    CUDA_CHECK(cudaMalloc(&d_floats, N_FLOAT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_min, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_floats, h_floats, N_FLOAT * sizeof(float), cudaMemcpyHostToDevice));

    float init_f = 1e30f;
    CUDA_CHECK(cudaMemcpy(d_min, &init_f, sizeof(float), cudaMemcpyHostToDevice));
    findMinFloat<<<(N_FLOAT + 255) / 256, 256>>>(d_floats, d_min, N_FLOAT);
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_min;
    CUDA_CHECK(cudaMemcpy(&gpu_min, d_min, sizeof(float), cudaMemcpyDeviceToHost));
    printf("CPU min: %.6f  |  GPU min: %.6f  |  Match: %s\n",
           true_min, gpu_min, (fabsf(gpu_min - true_min) < 1e-4f) ? "YES" : "NO");

    // Cleanup
    CUDA_CHECK(cudaFree(d_bytes)); CUDA_CHECK(cudaFree(d_hist));
    CUDA_CHECK(cudaFree(d_ints)); CUDA_CHECK(cudaFree(d_max));
    CUDA_CHECK(cudaFree(d_floats)); CUDA_CHECK(cudaFree(d_min));
    free(h_bytes); free(h_hist_naive); free(h_hist_priv);
    free(h_ints); free(h_floats);
    return 0;
}

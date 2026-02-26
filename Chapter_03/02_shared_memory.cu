/*
 * Chapter 03 — 02_shared_memory.cu
 *
 * Demonstrates shared memory as a programmer-managed cache.
 * Uses 1D stencil (running average) as the example because:
 *   - Each output element needs multiple neighboring input elements
 *   - Without shared memory, neighbors are re-loaded from slow global memory
 *   - With shared memory, each element is loaded once and reused
 *
 * Also demonstrates bank conflicts with a simple timing experiment.
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o shared_mem 02_shared_memory.cu
 * Run:
 *   ./shared_mem
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA Error at %s:%d — %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_err));          \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

const int BLOCK_SIZE = 256;
const int RADIUS     = 4;   // Stencil radius: average of 2*RADIUS+1 elements

// ================================================================
// NAIVE: Each thread reads its 2*RADIUS+1 neighbors from global memory.
// For RADIUS=4, that's 9 reads per thread — lots of redundant loads.
// Adjacent threads load overlapping regions.
// ================================================================
__global__ void stencilNaive(const float *in, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= RADIUS && i < n - RADIUS) {
        float sum = 0.0f;
        for (int j = -RADIUS; j <= RADIUS; j++)
            sum += in[i + j];  // 9 global memory reads per thread!
        out[i] = sum / (2 * RADIUS + 1);
    }
}

// ================================================================
// SHARED MEMORY: Load a tile (including halo) once into shared memory.
// Each element in global memory is loaded exactly once per block.
//
// Layout of shared memory tile:
//   [halo left | main data | halo right]
//    RADIUS     BLOCK_SIZE   RADIUS
// Total: BLOCK_SIZE + 2*RADIUS elements
// ================================================================
__global__ void stencilShared(const float *in, float *out, int n)
{
    // Shared memory tile: central block + halos on each side
    __shared__ float tile[BLOCK_SIZE + 2 * RADIUS];

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx  = threadIdx.x + RADIUS;  // offset in shared memory

    // --- Load main data ---
    if (globalIdx < n)
        tile[localIdx] = in[globalIdx];
    else
        tile[localIdx] = 0.0f;

    // --- Load left halo (first RADIUS threads load the left border) ---
    if (threadIdx.x < RADIUS) {
        int gi = globalIdx - RADIUS;
        tile[localIdx - RADIUS] = (gi >= 0) ? in[gi] : 0.0f;
    }

    // --- Load right halo (last RADIUS threads load the right border) ---
    if (threadIdx.x >= blockDim.x - RADIUS) {
        int gi = globalIdx + RADIUS;
        tile[localIdx + RADIUS] = (gi < n) ? in[gi] : 0.0f;
    }

    // --- Synchronize: all threads must finish loading before any compute ---
    __syncthreads();

    // --- Compute stencil from shared memory (fast on-chip reads) ---
    if (globalIdx >= RADIUS && globalIdx < n - RADIUS) {
        float sum = 0.0f;
        for (int j = -RADIUS; j <= RADIUS; j++)
            sum += tile[localIdx + j];  // fast shared memory reads!
        out[globalIdx] = sum / (2 * RADIUS + 1);
    }
}

// ================================================================
// BANK CONFLICT DEMONSTRATION
// Shared memory has 32 banks. Access to the same bank by multiple
// threads in a warp causes serialization.
//
// No conflict: thread i reads tile[i]     → bank = i % 32
// 2-way conflict: thread i reads tile[i*2] → banks 0,0,1,1,2,2,...
// 32-way conflict: thread i reads tile[i*32] → all threads hit bank 0!
// ================================================================
__global__ void bankConflictTest(float *output, int stride)
{
    __shared__ float smem[32 * 32];  // 32 banks * 32 entries

    int tid = threadIdx.x;
    // Write: no conflict (sequential)
    smem[tid] = (float)tid;
    __syncthreads();

    // Read: stride determines conflict level
    // stride=1: no conflict
    // stride=2: 2-way conflict
    // stride=32: 32-way conflict (worst case)
    output[tid] = smem[(tid * stride) % (32 * 32)];
}

// ================================================================
// Helper: time kernel and return ms
// ================================================================
float timeKernel(void (*f)(cudaStream_t), int reps = 10)
{
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    f(0); CUDA_CHECK(cudaDeviceSynchronize());  // warmup
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < reps; i++) f(0);
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaEventDestroy(s)); CUDA_CHECK(cudaEventDestroy(e));
    return ms / reps;
}

// ================================================================
// MAIN
// ================================================================
int main()
{
    const int N = 1 << 22;  // 4M elements
    size_t bytes = N * sizeof(float);

    printf("Stencil computation (radius=%d, N=%d)\n\n", RADIUS, N);

    float *h_in = (float*)malloc(bytes);
    float *h_out_naive = (float*)malloc(bytes);
    float *h_out_shared = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // --- Time naive (global memory) stencil ---
    auto naive_launch = [&](cudaStream_t s) {
        stencilNaive<<<blocks, BLOCK_SIZE, 0, s>>>(d_in, d_out, N);
    };
    float ms_naive = timeKernel(naive_launch);
    CUDA_CHECK(cudaMemcpy(h_out_naive, d_out, bytes, cudaMemcpyDeviceToHost));

    // --- Time shared memory stencil ---
    auto shared_launch = [&](cudaStream_t s) {
        stencilShared<<<blocks, BLOCK_SIZE, 0, s>>>(d_in, d_out, N);
    };
    float ms_shared = timeKernel(shared_launch);
    CUDA_CHECK(cudaMemcpy(h_out_shared, d_out, bytes, cudaMemcpyDeviceToHost));

    // --- Verify ---
    int errors = 0;
    for (int i = RADIUS; i < N - RADIUS; i++) {
        if (fabsf(h_out_naive[i] - h_out_shared[i]) > 0.01f) {
            printf("Mismatch at %d: naive=%f shared=%f\n", i,
                   h_out_naive[i], h_out_shared[i]);
            if (++errors > 5) break;
        }
    }
    if (errors == 0) printf("Verification PASSED.\n");

    printf("\n--- Stencil Performance ---\n");
    printf("  Naive (global memory):    %.3f ms\n", ms_naive);
    printf("  Shared memory:            %.3f ms\n", ms_shared);
    printf("  Speedup:                  %.2fx\n\n", ms_naive / ms_shared);

    // --- Bank conflict demo ---
    printf("--- Bank Conflict Test (32 threads) ---\n");
    float *d_bankout;
    CUDA_CHECK(cudaMalloc(&d_bankout, 32 * sizeof(float)));

    const char *conflict_labels[] = {
        "Stride=1 (no conflict)",
        "Stride=2 (2-way conflict)",
        "Stride=16 (16-way conflict)",
        "Stride=32 (32-way conflict / broadcast)"
    };
    int strides[] = {1, 2, 16, 32};

    for (int k = 0; k < 4; k++) {
        int stride = strides[k];
        auto bc_launch = [&](cudaStream_t s) {
            bankConflictTest<<<1, 32, 0, s>>>(d_bankout, stride);
        };
        float ms_bc = timeKernel(bc_launch, 1000);
        printf("  %-40s  %.4f us\n", conflict_labels[k], ms_bc * 1000.0f);
    }
    printf("\nNote: Stride=32 is a broadcast (all read same bank[0]) — no conflict!\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_bankout));
    free(h_in); free(h_out_naive); free(h_out_shared);
    return 0;
}

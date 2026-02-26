/*
 * Chapter 04 — 02_matmul_tiled.cu
 *
 * Tiled matrix multiplication using shared memory.
 * Key optimization: reduces global memory traffic by factor TILE_SIZE.
 *
 * Also includes a register-blocked variant for further speedup.
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o matmul_tiled 02_matmul_tiled.cu
 * Run:
 *   ./matmul_tiled
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
// TILED MATRIX MULTIPLICATION
// Block of TILE x TILE threads computes a TILE x TILE tile of C.
// Each tile iteration loads a TILE x TILE strip of A and B.
//
// Shared memory usage: 2 * TILE * TILE * 4 bytes = 8 KB for TILE=32
// ================================================================
template<int TILE>
__global__ void matmulTiled(const float *A, const float *B, float *C,
                             int M, int N, int K)
{
    // Shared memory tiles for A and B
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;  // global row of C
    int col = blockIdx.x * TILE + threadIdx.x;  // global col of C

    float acc = 0.0f;  // accumulator in register

    // Loop over tiles along K dimension
    int numTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; t++) {
        // --- Phase 1: Load tiles into shared memory ---
        // Thread (ty, tx) loads A[row][t*TILE + tx] and B[t*TILE + ty][col]

        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        // Boundary checks (handles K not divisible by TILE)
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
                                       ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
                                       ? B[b_row * N + col] : 0.0f;

        // --- Synchronize: all threads must finish loading ---
        __syncthreads();

        // --- Phase 2: Compute partial dot product from shared memory ---
        // All accesses hit fast on-chip shared memory
        #pragma unroll
        for (int k = 0; k < TILE; k++)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        // --- Synchronize: all threads must finish computing before next load ---
        __syncthreads();
    }

    // Write output
    if (row < M && col < N)
        C[row * N + col] = acc;
}

// ================================================================
// REGISTER-BLOCKED VARIANT
// Each thread computes a RB x RB sub-tile of C to increase
// arithmetic intensity further. TILE threads per dimension,
// each computing RB outputs.
//
// This is closer to how production GEMM kernels work.
// Thread (ty, tx) computes output block starting at
//   (by*TILE*RB + ty*RB, bx*TILE*RB + tx*RB)
// ================================================================
template<int TILE, int RB>
__global__ void matmulRegBlock(const float *A, const float *B, float *C,
                                int M, int N, int K)
{
    // Each block computes (TILE*RB) x (TILE*RB) output
    int base_row = blockIdx.y * (TILE * RB);
    int base_col = blockIdx.x * (TILE * RB);

    // Per-thread accumulator: RB x RB sub-tile in registers
    float acc[RB][RB] = {};

    // Shared memory for A and B tiles
    __shared__ float As[TILE * RB][TILE];
    __shared__ float Bs[TILE][TILE * RB];

    int numTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; t++) {
        // Load (TILE*RB) x TILE block of A
        for (int r = 0; r < RB; r++) {
            int gRow = base_row + threadIdx.y + r * TILE;
            int gCol = t * TILE + threadIdx.x;
            As[threadIdx.y + r * TILE][threadIdx.x] =
                (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
        }
        // Load TILE x (TILE*RB) block of B
        for (int c = 0; c < RB; c++) {
            int gRow = t * TILE + threadIdx.y;
            int gCol = base_col + threadIdx.x + c * TILE;
            Bs[threadIdx.y][threadIdx.x + c * TILE] =
                (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
        }

        __syncthreads();

        // Compute RB x RB output sub-tile
        for (int k = 0; k < TILE; k++)
            for (int r = 0; r < RB; r++)
                for (int c = 0; c < RB; c++)
                    acc[r][c] += As[threadIdx.y + r * TILE][k]
                               * Bs[k][threadIdx.x + c * TILE];

        __syncthreads();
    }

    // Write RB x RB results
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < RB; c++) {
            int gRow = base_row + threadIdx.y + r * TILE;
            int gCol = base_col + threadIdx.x + c * TILE;
            if (gRow < M && gCol < N)
                C[gRow * N + gCol] = acc[r][c];
        }
}

// ================================================================
// Helpers
// ================================================================
float timeKernelMs(void (*f)(cudaStream_t), int reps = 5)
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

void verifyResult(const float *C_gpu, const float *A, const float *B,
                  int M, int N, int K, int check_n = 256)
{
    // Verify a sample of output elements
    int errors = 0;
    srand(123);
    for (int t = 0; t < check_n; t++) {
        int r = rand() % M, c = rand() % N;
        float ref = 0.0f;
        for (int k = 0; k < K; k++)
            ref += A[r * K + k] * B[k * N + c];
        float diff = fabsf(C_gpu[r * N + c] - ref);
        if (diff > 1e-2f * fabsf(ref) + 1e-4f) {
            if (errors < 3)
                printf("  Error at (%d,%d): GPU=%.4f ref=%.4f diff=%.4f\n",
                       r, c, C_gpu[r*N+c], ref, diff);
            errors++;
        }
    }
    if (errors == 0)
        printf("  Verification PASSED (%d samples checked)\n", check_n);
    else
        printf("  Verification FAILED: %d/%d errors\n", errors, check_n);
}

// ================================================================
// MAIN
// ================================================================
int main()
{
    const int M = 2048, N = 2048, K = 2048;
    size_t bA = (size_t)M * K * sizeof(float);
    size_t bB = (size_t)K * N * sizeof(float);
    size_t bC = (size_t)M * N * sizeof(float);

    printf("Tiled Matrix Multiplication: %dx%d * %dx%d\n\n", M, K, K, N);

    float *h_A = (float*)malloc(bA);
    float *h_B = (float*)malloc(bB);
    float *h_C = (float*)malloc(bC);
    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX - 0.5f;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bA));
    CUDA_CHECK(cudaMalloc(&d_B, bB));
    CUDA_CHECK(cudaMalloc(&d_C, bC));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bB, cudaMemcpyHostToDevice));

    double ops = 2.0 * M * N * K;

    // ---- Tiled TILE=16 ----
    {
        constexpr int T = 16;
        dim3 block(T, T);
        dim3 grid((N + T - 1) / T, (M + T - 1) / T);
        auto f = [&](cudaStream_t s) {
            matmulTiled<T><<<grid, block, 0, s>>>(d_A, d_B, d_C, M, N, K);
        };
        float ms = timeKernelMs(f);
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bC, cudaMemcpyDeviceToHost));
        printf("Tiled TILE=16:\n");
        printf("  Time: %.3f ms   GFLOPS: %.1f\n", ms, ops / (ms * 1e-3) / 1e9);
        verifyResult(h_C, h_A, h_B, M, N, K);
        printf("  Shared mem per block: %.1f KB\n\n",
               2.0f * T * T * sizeof(float) / 1024);
    }

    // ---- Tiled TILE=32 ----
    {
        constexpr int T = 32;
        dim3 block(T, T);
        dim3 grid((N + T - 1) / T, (M + T - 1) / T);
        auto f = [&](cudaStream_t s) {
            matmulTiled<T><<<grid, block, 0, s>>>(d_A, d_B, d_C, M, N, K);
        };
        float ms = timeKernelMs(f);
        printf("Tiled TILE=32:\n");
        printf("  Time: %.3f ms   GFLOPS: %.1f\n", ms, ops / (ms * 1e-3) / 1e9);
        printf("  Shared mem per block: %.1f KB\n\n",
               2.0f * T * T * sizeof(float) / 1024);
    }

    // ---- Register blocked TILE=16, RB=2 ----
    {
        constexpr int T = 16, RB = 2;
        dim3 block(T, T);
        // Each block computes T*RB x T*RB output
        dim3 grid((N + T * RB - 1) / (T * RB), (M + T * RB - 1) / (T * RB));
        auto f = [&](cudaStream_t s) {
            matmulRegBlock<T, RB><<<grid, block, 0, s>>>(d_A, d_B, d_C, M, N, K);
        };
        float ms = timeKernelMs(f);
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bC, cudaMemcpyDeviceToHost));
        printf("Register-blocked (TILE=%d, RB=%d):\n", T, RB);
        printf("  Time: %.3f ms   GFLOPS: %.1f\n", ms, ops / (ms * 1e-3) / 1e9);
        verifyResult(h_C, h_A, h_B, M, N, K);
        printf("\n");
    }

    printf("RTX 4090 FP32 peak: ~82,600 GFLOPS\n");
    printf("Gap to peak is closed by Tensor Cores, software pipelining,\n");
    printf("vectorized loads, and other tricks used in cuBLAS.\n");

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);
    return 0;
}

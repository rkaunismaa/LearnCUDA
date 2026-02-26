/*
 * Chapter 12 — 03_tensor_cores.cu
 *
 * Demonstrates WMMA (Warp Matrix Multiply-Accumulate) API for Tensor Cores.
 * Implements a 16x16x16 FP16 matrix multiply using warp-level operations.
 *
 * Tensor Cores execute an entire 16x16x16 FP16 GEMM in a single warp
 * instruction, far more efficiently than CUDA core FP32 operations.
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o tensor_cores 03_tensor_cores.cu
 * Run:
 *   ./tensor_cores
 *
 * Note: WMMA requires CC >= 7.0 (Volta). RTX 4090 = CC 8.9 (Ada) — fully supported.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA Error at %s:%d — %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_err));          \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// WMMA tile size (16x16x16 is the fundamental size)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ================================================================
// FP32 CUDA core GEMM (for comparison)
// ================================================================
template<int TILE>
__global__ void matmulFP32(const float *A, const float *B, float *C,
                            int M, int N, int K)
{
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int ac = t * TILE + threadIdx.x, br = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && ac < K) ? A[row * K + ac] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (br < K && col < N) ? B[br * N + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// ================================================================
// FP16 WMMA GEMM
// Each warp computes a 16x16 tile of C = A * B using Tensor Cores.
// A is M x K (FP16, row-major)
// B is K x N (FP16, col-major — for WMMA col_major of B)
// C is M x N (FP32 accumulation)
//
// Grid: (N/16) x (M/16) blocks
// Block: (32, 1) threads = 1 warp per 16x16 output tile
// ================================================================
__global__ void matmulWMMA(const half *A, const half *B, float *C,
                            int M, int N, int K)
{
    // Which output tile does this warp compute?
    int warpRow = blockIdx.y * WMMA_M;  // row in C
    int warpCol = blockIdx.x * WMMA_N;  // col in C

    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major>    frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::col_major>    frag_b;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                   float>                    frag_c;

    // Initialize accumulator to zero
    wmma::fill_fragment(frag_c, 0.0f);

    // Iterate over K dimension in steps of WMMA_K
    for (int k = 0; k < K; k += WMMA_K) {
        if (warpRow < M && warpCol < N && k + WMMA_K <= K) {
            // Load 16x16 tile of A: starting at (warpRow, k)
            wmma::load_matrix_sync(frag_a, A + warpRow * K + k, K);

            // Load 16x16 tile of B: starting at (k, warpCol)
            // col_major: B is stored transposed, so stride is K
            wmma::load_matrix_sync(frag_b, B + k * N + warpCol, N);

            // Execute MMA: frag_c += frag_a * frag_b (Tensor Core instruction)
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }
    }

    // Store the 16x16 result to C
    if (warpRow < M && warpCol < N)
        wmma::store_matrix_sync(C + warpRow * N + warpCol, frag_c, N,
                                wmma::mem_row_major);
}

// ================================================================
// Timer
// ================================================================
float timeMs(void (*f)(cudaStream_t), int reps = 5)
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
    // Check for CC >= 7.0 (WMMA requirement)
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Tensor Core WMMA Demo\n");
    printf("GPU: %s (CC %d.%d)\n\n", prop.name, prop.major, prop.minor);

    if (prop.major < 7) {
        printf("Tensor Cores require CC >= 7.0. Your GPU (CC %d.%d) is not supported.\n",
               prop.major, prop.minor);
        return 1;
    }

    // Matrices must be multiples of WMMA_M/N/K (16)
    const int M = 2048, N = 2048, K = 2048;
    printf("Matrix size: A(%dx%d) * B(%dx%d) = C(%dx%d)\n\n", M, K, K, N, M, N);

    // Allocate FP32 host data
    float *h_A_fp32 = (float*)malloc(M * K * sizeof(float));
    float *h_B_fp32 = (float*)malloc(K * N * sizeof(float));
    srand(42);
    for (int i = 0; i < M * K; i++) h_A_fp32[i] = (float)(rand() % 100) / 100.0f - 0.5f;
    for (int i = 0; i < K * N; i++) h_B_fp32[i] = (float)(rand() % 100) / 100.0f - 0.5f;

    // Convert to FP16 for WMMA
    half *h_A_fp16 = (half*)malloc(M * K * sizeof(half));
    half *h_B_fp16 = (half*)malloc(K * N * sizeof(half));
    for (int i = 0; i < M * K; i++) h_A_fp16[i] = __float2half(h_A_fp32[i]);
    for (int i = 0; i < K * N; i++) h_B_fp16[i] = __float2half(h_B_fp32[i]);

    // Device memory
    float *d_A_fp32, *d_B_fp32, *d_C_fp32;  // For FP32 kernel
    half  *d_A_fp16, *d_B_fp16;             // For WMMA kernel
    float *d_C_wmma;                         // WMMA accumulates in FP32

    CUDA_CHECK(cudaMalloc(&d_A_fp32, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_fp32, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_fp32, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A_fp16, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B_fp16, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C_wmma, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A_fp32, h_A_fp32, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_fp32, h_B_fp32, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_fp16, h_A_fp16, M * K * sizeof(half),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_fp16, h_B_fp16, K * N * sizeof(half),  cudaMemcpyHostToDevice));

    double ops = 2.0 * M * N * K;

    // ---- FP32 Tiled GEMM ----
    {
        constexpr int T = 16;
        dim3 block(T, T), grid((N + T-1)/T, (M + T-1)/T);
        auto f = [&](cudaStream_t s) {
            matmulFP32<T><<<grid, block, 0, s>>>(d_A_fp32, d_B_fp32, d_C_fp32, M, N, K);
        };
        float ms = timeMs(f);
        printf("FP32 Tiled GEMM (CUDA cores):  %.3f ms  =  %.0f GFLOPS\n",
               ms, ops / (ms * 1e-3) / 1e9);
    }

    // ---- FP16 WMMA GEMM (Tensor Cores) ----
    {
        // One warp (32 threads) per 16x16 output tile
        // Grid: (N/WMMA_N, M/WMMA_M)
        dim3 block(32, 1);  // 1 warp = 32 threads
        dim3 grid(N / WMMA_N, M / WMMA_M);
        auto f = [&](cudaStream_t s) {
            matmulWMMA<<<grid, block, 0, s>>>(d_A_fp16, d_B_fp16, d_C_wmma, M, N, K);
        };
        float ms = timeMs(f);
        printf("FP16 WMMA GEMM (Tensor Cores): %.3f ms  =  %.0f GFLOPS\n",
               ms, ops / (ms * 1e-3) / 1e9);
        printf("\nRTX 4090 theoretical peaks:\n");
        printf("  FP32 CUDA cores: ~82,600 GFLOPS\n");
        printf("  FP16 Tensor Core: ~330,000 GFLOPS\n");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_A_fp32)); CUDA_CHECK(cudaFree(d_B_fp32)); CUDA_CHECK(cudaFree(d_C_fp32));
    CUDA_CHECK(cudaFree(d_A_fp16)); CUDA_CHECK(cudaFree(d_B_fp16)); CUDA_CHECK(cudaFree(d_C_wmma));
    free(h_A_fp32); free(h_B_fp32); free(h_A_fp16); free(h_B_fp16);
    return 0;
}

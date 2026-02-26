/*
 * Chapter 04 — 01_matmul_naive.cu
 *
 * Naive matrix multiplication: C = A * B
 * One thread per output element, direct global memory access.
 *
 * This is the baseline we'll optimize in 02_matmul_tiled.cu
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o matmul_naive 01_matmul_naive.cu
 * Run:
 *   ./matmul_naive
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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
// NAIVE KERNEL
// One thread computes one element of C = A * B.
// A is M x K, B is K x N, C is M x N.
// Row-major (C-order) storage.
// ================================================================
__global__ void matmulNaive(const float *A, const float *B, float *C,
                             int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // A[row][k] = A[row * K + k]  — coalesced for fixed row
            // B[k][col] = B[k * N + col]  — NOT coalesced (stride N apart)
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ================================================================
// CPU reference for verification (only run on small matrices)
// ================================================================
void matmulCPU(const float *A, const float *B, float *C, int M, int N, int K)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[m * K + k] * B[k * N + n];
            C[m * N + n] = sum;
        }
}

// ================================================================
// Timer using CUDA events
// ================================================================
float timeKernelMs(void (*f)(cudaStream_t), int reps = 5)
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
    // --- Matrix dimensions ---
    const int M = 2048, N = 2048, K = 2048;
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    printf("Matrix Multiplication: A(%dx%d) * B(%dx%d) = C(%dx%d)\n\n",
           M, K, K, N, M, N);

    // --- Host allocation and initialization ---
    float *h_A = (float*)malloc(bytesA);
    float *h_B = (float*)malloc(bytesB);
    float *h_C = (float*)malloc(bytesC);

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX - 0.5f;

    // --- Device allocation ---
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    // --- Launch configuration ---
    // 16x16 thread blocks are standard for matrix multiply
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    printf("Launch config: grid(%d,%d), block(%d,%d)\n\n",
           grid.x, grid.y, block.x, block.y);

    // --- Time the kernel ---
    auto launch = [&](cudaStream_t s) {
        matmulNaive<<<grid, block, 0, s>>>(d_A, d_B, d_C, M, N, K);
    };
    float ms = timeKernelMs(launch, 5);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

    // --- Performance metrics ---
    double gflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e9;
    printf("Naive matmul time:  %.3f ms\n", ms);
    printf("Performance:        %.1f GFLOPS\n", gflops);
    printf("(RTX 4090 FP32 peak: ~82,000 GFLOPS = 82 TFLOPS)\n\n");

    // --- Verify against CPU on a small portion ---
    printf("Verifying first 8x8 submatrix against CPU...\n");
    const int VSIZE = 8;
    float *h_A_v = (float*)malloc(VSIZE * VSIZE * sizeof(float));
    float *h_B_v = (float*)malloc(VSIZE * VSIZE * sizeof(float));
    float *h_C_cpu = (float*)malloc(VSIZE * VSIZE * sizeof(float));
    float *h_C_gpu = (float*)malloc(VSIZE * VSIZE * sizeof(float));

    // Extract top-left VSIZE x VSIZE submatrices
    for (int i = 0; i < VSIZE; i++)
        for (int j = 0; j < VSIZE; j++) {
            h_A_v[i * VSIZE + j] = h_A[i * K + j];
            h_B_v[i * VSIZE + j] = h_B[i * N + j];
            h_C_gpu[i * VSIZE + j] = h_C[i * N + j];
        }
    matmulCPU(h_A_v, h_B_v, h_C_cpu, VSIZE, VSIZE, VSIZE);

    float maxErr = 0.0f;
    for (int i = 0; i < VSIZE * VSIZE; i++)
        maxErr = fmaxf(maxErr, fabsf(h_C_gpu[i] - h_C_cpu[i]));
    printf("Max error (top-left %dx%d): %.6f\n", VSIZE, VSIZE, maxErr);

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);
    free(h_A_v); free(h_B_v); free(h_C_cpu); free(h_C_gpu);

    return 0;
}

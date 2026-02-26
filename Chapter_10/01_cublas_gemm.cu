/*
 * Chapter 10 — 01_cublas_gemm.cu
 *
 * Demonstrates cuBLAS for matrix multiplication (SGEMM).
 * Compares our Chapter 04 tiled kernel vs cuBLAS for GFLOPS.
 * Also demonstrates cublasSaxpy (AXPY) and cublasSdot (dot product).
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o cublas_gemm 01_cublas_gemm.cu -lcublas
 * Run:
 *   ./cublas_gemm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA Error at %s:%d — %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_err));          \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t _s = (call);                                         \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS Error at %s:%d — code %d\n",          \
                    __FILE__, __LINE__, (int)_s);                           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Our tiled matmul from Chapter 04 (for comparison)
template<int TILE>
__global__ void matmulTiled(const float *A, const float *B, float *C,
                             int M, int N, int K)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

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
    const int M = 2048, N = 2048, K = 2048;
    size_t bA = (size_t)M * K * sizeof(float);
    size_t bB = (size_t)K * N * sizeof(float);
    size_t bC = (size_t)M * N * sizeof(float);

    printf("cuBLAS SGEMM: A(%dx%d) * B(%dx%d) = C(%dx%d)\n\n", M, K, K, N, M, N);

    float *h_A = (float*)malloc(bA);
    float *h_B = (float*)malloc(bB);
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

    // ---- Our tiled matmul ----
    {
        constexpr int T = 16;
        dim3 block(T, T), grid((N + T-1)/T, (M + T-1)/T);
        auto f = [&](cudaStream_t s) {
            matmulTiled<T><<<grid, block, 0, s>>>(d_A, d_B, d_C, M, N, K);
        };
        float ms = timeMs(f);
        printf("Our tiled matmul (TILE=16): %.3f ms  =  %.1f GFLOPS\n",
               ms, ops / (ms * 1e-3) / 1e9);
    }

    // ---- cuBLAS SGEMM ----
    // Note on row-major trick:
    // cuBLAS expects column-major. For row-major A (M×K) and B (K×N):
    // We compute C = B * A using cuBLAS (swapped order), which gives
    // the result equivalent to A * B in row-major storage.
    {
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));

        float alpha = 1.0f, beta = 0.0f;

        // cuBLAS computes C_cm = alpha * op(A_cm) * op(B_cm) + beta * C_cm
        // For row-major C = A * B:
        //   Treat A (M×K rowmaj) as Aᵀ (K×M colmaj)
        //   Treat B (K×N rowmaj) as Bᵀ (N×K colmaj)
        //   Compute Cᵀ (N×M colmaj) = Bᵀ * Aᵀ  →  cublasSgemm with B first
        auto f = [&](cudaStream_t s) {
            cublasSetStream(handle, s);
            CUBLAS_CHECK(cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,          // n, m, k (note swapped N,M!)
                &alpha,
                d_B, N,           // B, leading dim of B (columns = N)
                d_A, K,           // A, leading dim of A (columns = K)
                &beta,
                d_C, N));         // C, leading dim of C (columns = N)
        };
        float ms = timeMs(f);
        printf("cuBLAS cublasSgemm:         %.3f ms  =  %.1f GFLOPS\n",
               ms, ops / (ms * 1e-3) / 1e9);

        CUBLAS_CHECK(cublasDestroy(handle));
    }

    // ---- Level 1: AXPY and DOT ----
    printf("\n--- Level 1 Operations ---\n");
    {
        const int VN = 1 << 20;  // 1M floats
        float *d_x, *d_y;
        CUDA_CHECK(cudaMalloc(&d_x, VN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, VN * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_x, 0, VN * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_y, 0, VN * sizeof(float)));

        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));

        // y = alpha * x + y  (AXPY)
        float alpha = 2.0f;
        CUBLAS_CHECK(cublasSaxpy(handle, VN, &alpha, d_x, 1, d_y, 1));

        // dot = x · y  (DOT)
        float dot_result;
        CUBLAS_CHECK(cublasSdot(handle, VN, d_x, 1, d_y, 1, &dot_result));
        printf("cublasSdot(zeros, zeros): %.4f (expected 0)\n", dot_result);

        // nrm2 = ||x||_2  (Euclidean norm)
        float norm;
        CUBLAS_CHECK(cublasSnrm2(handle, VN, d_x, 1, &norm));
        printf("cublasSnrm2(zeros): %.4f (expected 0)\n", norm);

        CUBLAS_CHECK(cublasDestroy(handle));
        CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    }

    printf("\nNote: cuBLAS uses Tensor Cores on RTX 4090, achieving far higher\n");
    printf("GFLOPS than our FP32-only kernel. Use CUBLAS_TF32_TENSOR_OP_MATH\n");
    printf("for automatic TF32/FP16 Tensor Core usage.\n");

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B);
    return 0;
}

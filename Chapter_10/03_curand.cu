/*
 * Chapter 10 — 03_curand.cu
 *
 * Demonstrates cuRAND for GPU random number generation:
 *   - Uniform distribution
 *   - Normal (Gaussian) distribution
 *   - Generation throughput benchmark
 *   - Statistical verification of generated values
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o curand_demo 03_curand.cu -lcurand
 * Run:
 *   ./curand_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA Error at %s:%d — %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_err));          \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CURAND_CHECK(call)                                                  \
    do {                                                                    \
        curandStatus_t _s = (call);                                         \
        if (_s != CURAND_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuRAND Error at %s:%d — code %d\n",          \
                    __FILE__, __LINE__, (int)_s);                           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

struct Square { __device__ float operator()(float x) const { return x * x; } };
struct Diff   { float m; __device__ float operator()(float x) const { return (x-m)*(x-m); } };

// Compute mean and variance of a device float array
void computeStats(const float *d_data, int n, float *mean, float *stddev)
{
    thrust::device_ptr<const float> ptr(d_data);
    float sum = thrust::reduce(ptr, ptr + n, 0.0f, thrust::plus<float>());
    *mean = sum / n;
    float sum_sq_diff = thrust::transform_reduce(ptr, ptr + n,
                                                  Diff{*mean}, 0.0f,
                                                  thrust::plus<float>());
    *stddev = sqrtf(sum_sq_diff / n);
}

int main()
{
    printf("cuRAND Random Number Generation\n\n");

    // Create generator
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 12345ULL));

    // ================================================================
    // 1. Uniform distribution U[0,1)
    // ================================================================
    printf("--- Uniform Distribution U[0, 1) ---\n");
    {
        const int N = 1 << 22;  // 4M floats
        float *d_uniform;
        CUDA_CHECK(cudaMalloc(&d_uniform, N * sizeof(float)));

        // Generate
        cudaEvent_t s, e;
        CUDA_CHECK(cudaEventCreate(&s));
        CUDA_CHECK(cudaEventCreate(&e));
        CUDA_CHECK(cudaEventRecord(s));
        CURAND_CHECK(curandGenerateUniform(gen, d_uniform, N));
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));

        float mean, stddev;
        computeStats(d_uniform, N, &mean, &stddev);

        printf("  N=%d  time=%.3f ms  %.1f GB/s\n",
               N, ms, N * 4.0f / (ms * 1e-3f) / 1e9f);
        printf("  Mean:   %.6f (expected 0.5)\n", mean);
        printf("  StdDev: %.6f (expected %.6f)\n\n", stddev, sqrtf(1.0f/12.0f));

        CUDA_CHECK(cudaEventDestroy(s));
        CUDA_CHECK(cudaEventDestroy(e));
        CUDA_CHECK(cudaFree(d_uniform));
    }

    // ================================================================
    // 2. Normal distribution N(mu, sigma)
    // ================================================================
    printf("--- Normal Distribution N(0, 1) ---\n");
    {
        const int N = 1 << 22;
        float *d_normal;
        CUDA_CHECK(cudaMalloc(&d_normal, N * sizeof(float)));

        // N must be even for curandGenerateNormal
        CURAND_CHECK(curandGenerateNormal(gen, d_normal, N, 0.0f, 1.0f));
        CUDA_CHECK(cudaDeviceSynchronize());

        float mean, stddev;
        computeStats(d_normal, N, &mean, &stddev);
        printf("  Mean:   %.6f (expected 0)\n", mean);
        printf("  StdDev: %.6f (expected 1)\n\n", stddev);

        CUDA_CHECK(cudaFree(d_normal));
    }

    // ================================================================
    // 3. Throughput scaling
    // ================================================================
    printf("--- Generation Throughput (uniform) ---\n");
    printf("%-12s  %-12s  %-12s\n", "N", "Time (ms)", "GB/s");
    printf("%s\n", std::string(40, '-').c_str());

    size_t sizes[] = {1<<20, 1<<22, 1<<24, 1<<26};
    const char *labels[] = {"1M", "4M", "16M", "64M"};
    for (int s = 0; s < 4; s++) {
        size_t N = sizes[s];
        float *d_data;
        CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));

        cudaEvent_t ts, te;
        CUDA_CHECK(cudaEventCreate(&ts));
        CUDA_CHECK(cudaEventCreate(&te));
        CURAND_CHECK(curandGenerateUniform(gen, d_data, N));  // warmup
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(ts));
        CURAND_CHECK(curandGenerateUniform(gen, d_data, N));
        CUDA_CHECK(cudaEventRecord(te));
        CUDA_CHECK(cudaEventSynchronize(te));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, ts, te));

        printf("%-12s  %-12.3f  %-12.1f\n",
               labels[s], ms, N * 4.0f / (ms * 1e-3f) / 1e9f);

        CUDA_CHECK(cudaEventDestroy(ts)); CUDA_CHECK(cudaEventDestroy(te));
        CUDA_CHECK(cudaFree(d_data));
    }

    // ================================================================
    // 4. Different generator types
    // ================================================================
    printf("\n--- Generator Types ---\n");
    const int VN = 1 << 20;
    float *d_test;
    CUDA_CHECK(cudaMalloc(&d_test, VN * sizeof(float)));

    curandRngType_t types[] = {
        CURAND_RNG_PSEUDO_DEFAULT,
        CURAND_RNG_PSEUDO_MT19937,
        CURAND_RNG_PSEUDO_MRG32K3A,
        CURAND_RNG_QUASI_DEFAULT
    };
    const char *type_names[] = {"XORWOW (default)", "MT19937", "MRG32K3A", "Sobol (quasi)"};

    for (int t = 0; t < 4; t++) {
        curandGenerator_t gen2;
        CURAND_CHECK(curandCreateGenerator(&gen2, types[t]));
        if (types[t] != CURAND_RNG_QUASI_DEFAULT)
            CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen2, 42));

        cudaEvent_t ts, te;
        CUDA_CHECK(cudaEventCreate(&ts));
        CUDA_CHECK(cudaEventCreate(&te));
        CUDA_CHECK(cudaEventRecord(ts));
        CURAND_CHECK(curandGenerateUniform(gen2, d_test, VN));
        CUDA_CHECK(cudaEventRecord(te));
        CUDA_CHECK(cudaEventSynchronize(te));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, ts, te));

        printf("  %-20s  %.3f ms\n", type_names[t], ms);
        CUDA_CHECK(cudaEventDestroy(ts)); CUDA_CHECK(cudaEventDestroy(te));
        CURAND_CHECK(curandDestroyGenerator(gen2));
    }

    CURAND_CHECK(curandDestroyGenerator(gen));
    CUDA_CHECK(cudaFree(d_test));
    return 0;
}

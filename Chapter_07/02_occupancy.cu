/*
 * Chapter 07 — 02_occupancy.cu
 *
 * Demonstrates SM occupancy analysis:
 *   - cudaOccupancyMaxActiveBlocksPerMultiprocessor
 *   - cudaOccupancyMaxPotentialBlockSize (auto-select optimal block size)
 *   - __launch_bounds__ to control register usage
 *   - Benchmarking the same kernel at different block sizes
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o occupancy 02_occupancy.cu --ptxas-options=-v
 *   (--ptxas-options=-v prints register usage per kernel)
 * Run:
 *   ./occupancy
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

const int N = 1 << 24;  // 16M floats

// ================================================================
// Basic kernel — compiler chooses register count
// ================================================================
__global__ void copyKernel(const float *src, float *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] * 1.5f + 0.001f;
}

// ================================================================
// Same kernel with __launch_bounds__
// Tells the compiler: max 256 threads per block, min 4 blocks per SM
// This allows the compiler to make better register allocation decisions.
// ================================================================
__launch_bounds__(256, 4)
__global__ void copyKernelBounded(const float *src, float *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] * 1.5f + 0.001f;
}

// ================================================================
// Heavy register kernel — uses many local variables
// Will likely have lower occupancy due to register pressure
// ================================================================
__global__ void heavyKernel(const float *src, float *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Many independent local variables → high register usage
    float a = src[i];
    float b = a * a;
    float c = b + a;
    float d = c * b;
    float e = d - c;
    float f = e * e + b;
    float g = f / (a + 1.0f);
    float h2 = g * c + d;
    float ii = h2 - e * f;
    float j = ii + g * g;
    float k2 = j / (h2 + 1.0f);
    float l = k2 * ii + j;
    dst[i] = l;
}

// ================================================================
// Query and print occupancy for a kernel at a given block size
// ================================================================
template<typename F>
void printOccupancy(const char *name, F kernel, int blockSize, int sharedBytes = 0)
{
    int activeBlocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocks, kernel, blockSize, sharedBytes));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    float occupancy = (float)(activeBlocks * blockSize) / prop.maxThreadsPerMultiProcessor;
    printf("  %-35s blockSize=%4d  activBlocks/SM=%2d  occupancy=%5.1f%%\n",
           name, blockSize, activeBlocks, occupancy * 100.0f);
}

// ================================================================
// Time kernel at different block sizes, print bandwidth
// ================================================================
void benchmarkBlockSizes(float *d_src, float *d_dst, int n)
{
    size_t bytes = (size_t)n * sizeof(float);

    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));

    printf("\n  %-10s  %-10s  %-15s  %-12s\n", "BlockSize", "Blocks", "Time(ms)", "BW(GB/s)");
    printf("  %s\n", std::string(55, '-').c_str());

    for (int bs : {32, 64, 128, 256, 512, 1024}) {
        int blocks = (n + bs - 1) / bs;

        // Warmup
        copyKernel<<<blocks, bs>>>(d_src, d_dst, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(s));
        for (int r = 0; r < 10; r++)
            copyKernel<<<blocks, bs>>>(d_src, d_dst, n);
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        ms /= 10;

        float bw = (2.0f * bytes) / (ms * 1e-3f) / 1e9f;
        printf("  %-10d  %-10d  %-15.3f  %-12.1f\n", bs, blocks, ms, bw);
    }

    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
}

int main()
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("Occupancy Analysis on %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Max threads/SM: %d  |  Warp size: %d\n\n",
           prop.maxThreadsPerMultiProcessor, prop.warpSize);

    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_src, 0, N * sizeof(float)));

    // ================================================================
    // 1. Occupancy at various block sizes
    // ================================================================
    printf("--- Occupancy Query (cudaOccupancyMaxActiveBlocksPerMultiprocessor) ---\n");
    for (int bs : {32, 64, 128, 256, 512, 1024}) {
        printOccupancy("copyKernel", copyKernel, bs);
    }
    printf("\n");
    for (int bs : {32, 64, 128, 256}) {
        printOccupancy("heavyKernel (high reg)", heavyKernel, bs);
    }

    // ================================================================
    // 2. Auto-select optimal block size
    // ================================================================
    printf("\n--- Auto-select with cudaOccupancyMaxPotentialBlockSize ---\n");
    {
        int minGridSize, blockSize;
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, &blockSize, copyKernel, 0, 0));
        printf("copyKernel:  optimal blockSize=%d, minGridSize=%d\n",
               blockSize, minGridSize);
    }
    {
        int minGridSize, blockSize;
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, &blockSize, heavyKernel, 0, 0));
        printf("heavyKernel: optimal blockSize=%d, minGridSize=%d\n",
               blockSize, minGridSize);
    }

    // ================================================================
    // 3. __launch_bounds__ effect
    // ================================================================
    printf("\n--- __launch_bounds__ Occupancy ---\n");
    for (int bs : {64, 128, 256}) {
        printOccupancy("copyKernel (no bounds)", copyKernel, bs);
        printOccupancy("copyKernelBounded(256,4)", copyKernelBounded, bs);
    }

    // ================================================================
    // 4. Benchmark at different block sizes
    // ================================================================
    printf("\n--- Bandwidth at Different Block Sizes (copyKernel) ---\n");
    benchmarkBlockSizes(d_src, d_dst, N);

    printf("\nConclusion: For memory-bound kernels, occupancy above ~50%%\n");
    printf("is usually sufficient. Block sizes of 128-256 are typical sweet spots.\n");

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    return 0;
}

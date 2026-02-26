/*
 * Chapter 01 — 01_hello_cuda.cu
 *
 * The classic "Hello, World!" for CUDA.
 * Demonstrates:
 *   - Writing a __global__ kernel function
 *   - Launching a kernel with <<<grid, block>>> syntax
 *   - Using threadIdx and blockIdx to identify each thread
 *   - cudaDeviceSynchronize() to wait for GPU work to complete
 *
 * Compile:
 *   nvcc -o hello 01_hello_cuda.cu
 * Run:
 *   ./hello
 */

#include <stdio.h>

// ============================================================
// CUDA KERNEL
// A kernel is a function that runs on the GPU.
// __global__ means: "callable from host, runs on device"
// Every thread executes this function independently.
// ============================================================
__global__ void helloKernel()
{
    // threadIdx.x : thread's index within its block (0..blockDim.x-1)
    // blockIdx.x  : block's index within the grid   (0..gridDim.x-1)
    // blockDim.x  : number of threads per block
    printf("Hello from GPU! Block %d, Thread %d\n",
           blockIdx.x, threadIdx.x);
}

// ============================================================
// HOST (CPU) MAIN
// ============================================================
int main()
{
    printf("Hello from CPU!\n\n");

    // Launch configuration:
    //   - 2 blocks in the grid
    //   - 4 threads per block
    // Total threads = 2 * 4 = 8
    int numBlocks  = 2;
    int numThreads = 4;

    // <<<gridDim, blockDim>>> is CUDA's kernel launch syntax.
    // This is not standard C/C++ — nvcc processes it.
    helloKernel<<<numBlocks, numThreads>>>();

    // The kernel launch is ASYNCHRONOUS. The CPU continues
    // immediately while the GPU runs. We must synchronize
    // to ensure the GPU has finished before we exit.
    cudaDeviceSynchronize();

    printf("\nAll GPU threads finished.\n");
    printf("Notice: GPU thread output order is non-deterministic!\n");

    return 0;
}

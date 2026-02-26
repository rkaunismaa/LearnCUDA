/*
 * Chapter 02 — 02_thread_indexing.cu
 *
 * Demonstrates 1D, 2D, and 3D thread indexing patterns.
 * Also introduces the grid-stride loop pattern.
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o indexing 02_thread_indexing.cu
 * Run:
 *   ./indexing
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

// ================================================================
// 1D indexing: fill array with its own global index
// ================================================================
__global__ void fill1D(int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = i;
}

// ================================================================
// 2D indexing: fill a W*H matrix where each element = row*100 + col
// ================================================================
__global__ void fill2D(int *out, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = row * width + col;  // row-major linearization
        out[idx] = row * 100 + col;
    }
}

// ================================================================
// 3D indexing: fill a X*Y*Z volume where each element = z*10000 + y*100 + x
// ================================================================
__global__ void fill3D(int *out, int Nx, int Ny, int Nz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int idx = z * (Nx * Ny) + y * Nx + x;
        out[idx] = z * 10000 + y * 100 + x;
    }
}

// ================================================================
// Grid-stride loop: robust 1D pattern that works for any array size
// regardless of grid size. Each thread processes multiple elements.
// ================================================================
__global__ void fillGridStride(int *out, int n)
{
    // Total number of threads in the grid
    int stride = gridDim.x * blockDim.x;

    // Starting index for this thread
    int start = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes elements spaced by 'stride'
    for (int i = start; i < n; i += stride)
        out[i] = i;
}

// ================================================================
// Helper: print a small array
// ================================================================
void printArray(const char *label, int *arr, int n)
{
    printf("%s: [", label);
    int show = (n > 16) ? 16 : n;
    for (int i = 0; i < show; i++)
        printf("%d%s", arr[i], (i < show - 1) ? ", " : "");
    if (n > show) printf(", ...");
    printf("]\n");
}

// ================================================================
// MAIN
// ================================================================
int main()
{
    // --- 1D example ---
    printf("=== 1D Thread Indexing ===\n");
    {
        const int N = 20;
        int *h_out = (int*)malloc(N * sizeof(int));
        int *d_out;
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));

        fill1D<<<1, 32>>>(d_out, N);  // 1 block of 32 threads, only 20 needed
        CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

        printArray("fill1D (20 elements, 32 threads)", h_out, N);
        CUDA_CHECK(cudaFree(d_out));
        free(h_out);
    }

    // --- 2D example ---
    printf("\n=== 2D Thread Indexing ===\n");
    {
        const int W = 5, H = 4;
        int *h_out = (int*)malloc(W * H * sizeof(int));
        int *d_out;
        CUDA_CHECK(cudaMalloc(&d_out, W * H * sizeof(int)));

        dim3 block(4, 4);
        dim3 grid((W + 3) / 4, (H + 3) / 4);
        fill2D<<<grid, block>>>(d_out, W, H);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, W * H * sizeof(int), cudaMemcpyDeviceToHost));

        printf("fill2D (%dx%d matrix, values = row*100+col):\n", W, H);
        for (int r = 0; r < H; r++) {
            printf("  Row %d: ", r);
            for (int c = 0; c < W; c++)
                printf("%4d ", h_out[r * W + c]);
            printf("\n");
        }
        CUDA_CHECK(cudaFree(d_out));
        free(h_out);
    }

    // --- 3D example ---
    printf("\n=== 3D Thread Indexing ===\n");
    {
        const int Nx = 4, Ny = 3, Nz = 2;
        int total = Nx * Ny * Nz;
        int *h_out = (int*)malloc(total * sizeof(int));
        int *d_out;
        CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(int)));

        dim3 block(4, 4, 2);
        dim3 grid(1, 1, 1);
        fill3D<<<grid, block>>>(d_out, Nx, Ny, Nz);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, total * sizeof(int), cudaMemcpyDeviceToHost));

        printf("fill3D (%dx%dx%d volume, values = z*10000+y*100+x):\n", Nx, Ny, Nz);
        for (int z = 0; z < Nz; z++) {
            printf("  Slice z=%d:\n", z);
            for (int y = 0; y < Ny; y++) {
                printf("    y=%d: ", y);
                for (int x = 0; x < Nx; x++)
                    printf("%6d ", h_out[z * Nx * Ny + y * Nx + x]);
                printf("\n");
            }
        }
        CUDA_CHECK(cudaFree(d_out));
        free(h_out);
    }

    // --- Grid-stride loop example ---
    printf("\n=== Grid-Stride Loop ===\n");
    {
        const int N = 50;
        int *h_out = (int*)malloc(N * sizeof(int));
        int *d_out;
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));

        // Only 2 blocks of 8 threads = 16 threads total, but N=50
        // Each thread processes ceil(50/16) elements via the stride loop
        fillGridStride<<<2, 8>>>(d_out, N);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

        printf("fillGridStride (50 elements, only 16 threads):\n");
        printArray("  Result", h_out, N);

        CUDA_CHECK(cudaFree(d_out));
        free(h_out);
    }

    printf("\nDone.\n");
    return 0;
}

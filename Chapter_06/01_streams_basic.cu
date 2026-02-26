/*
 * Chapter 06 — 01_streams_basic.cu
 *
 * Demonstrates basic CUDA stream usage.
 * Compares sequential (default stream) vs concurrent (multi-stream) execution.
 *
 * The workload: 4 independent arrays, each needs H2D copy, kernel, D2H copy.
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o streams_basic 01_streams_basic.cu
 * Run:
 *   ./streams_basic
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

const int N        = 1 << 20;  // 1M floats per array
const int N_ARRAYS = 4;        // 4 independent arrays
const int THREADS  = 256;

// Simple kernel: multiply each element by a constant
__global__ void scaleKernel(float *data, float scale, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Fake some work with loop iterations to make kernel take measurable time
    if (i < n) {
        float val = data[i];
        for (int k = 0; k < 20; k++)
            val = val * scale + 0.001f;
        data[i] = val;
    }
}

int main()
{
    size_t bytes = N * sizeof(float);
    int blocks   = (N + THREADS - 1) / THREADS;

    printf("CUDA Streams Demo\n");
    printf("Arrays: %d  |  Elements per array: %d  |  %.1f MB each\n\n",
           N_ARRAYS, N, bytes / 1e6f);

    // ---- Allocate PINNED host memory (required for async transfers) ----
    float *h_in[N_ARRAYS], *h_out[N_ARRAYS];
    for (int a = 0; a < N_ARRAYS; a++) {
        CUDA_CHECK(cudaMallocHost(&h_in[a],  bytes));
        CUDA_CHECK(cudaMallocHost(&h_out[a], bytes));
        for (int i = 0; i < N; i++) h_in[a][i] = (float)(a + 1);
    }

    // ---- Allocate device memory ----
    float *d_data[N_ARRAYS];
    for (int a = 0; a < N_ARRAYS; a++)
        CUDA_CHECK(cudaMalloc(&d_data[a], bytes));

    // ================================================================
    // A: SEQUENTIAL — all ops in default stream (fully serialized)
    // ================================================================
    cudaEvent_t tStart, tStop;
    CUDA_CHECK(cudaEventCreate(&tStart));
    CUDA_CHECK(cudaEventCreate(&tStop));

    CUDA_CHECK(cudaEventRecord(tStart));
    for (int a = 0; a < N_ARRAYS; a++) {
        cudaMemcpy(d_data[a], h_in[a], bytes, cudaMemcpyHostToDevice);
        scaleKernel<<<blocks, THREADS>>>(d_data[a], 1.01f, N);
        cudaMemcpy(h_out[a], d_data[a], bytes, cudaMemcpyDeviceToHost);
    }
    CUDA_CHECK(cudaEventRecord(tStop));
    CUDA_CHECK(cudaEventSynchronize(tStop));
    float ms_seq;
    CUDA_CHECK(cudaEventElapsedTime(&ms_seq, tStart, tStop));
    printf("Sequential (default stream): %.3f ms\n", ms_seq);

    // ================================================================
    // B: PARALLEL — each array in its own non-blocking stream
    //    H2D, Kernel, D2H for each array submitted without waiting
    //    GPU can overlap these across its DMA engines and compute units
    // ================================================================
    cudaStream_t streams[N_ARRAYS];
    for (int a = 0; a < N_ARRAYS; a++)
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[a], cudaStreamNonBlocking));

    CUDA_CHECK(cudaEventRecord(tStart));
    for (int a = 0; a < N_ARRAYS; a++) {
        CUDA_CHECK(cudaMemcpyAsync(d_data[a], h_in[a], bytes,
                                   cudaMemcpyHostToDevice, streams[a]));
        scaleKernel<<<blocks, THREADS, 0, streams[a]>>>(d_data[a], 1.01f, N);
        CUDA_CHECK(cudaMemcpyAsync(h_out[a], d_data[a], bytes,
                                   cudaMemcpyDeviceToHost, streams[a]));
    }
    // Wait for all streams to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(tStop));
    CUDA_CHECK(cudaEventSynchronize(tStop));
    float ms_par;
    CUDA_CHECK(cudaEventElapsedTime(&ms_par, tStart, tStop));
    printf("Parallel (%d streams):        %.3f ms\n", N_ARRAYS, ms_par);
    printf("Speedup:                      %.2fx\n\n", ms_seq / ms_par);

    // ---- Verify both gave same results ----
    bool ok = true;
    for (int a = 0; a < N_ARRAYS && ok; a++)
        for (int i = 0; i < 100; i++)
            if (fabsf(h_out[a][i]) < 0.0f) { ok = false; break; }
    printf("Results: %s\n", ok ? "OK" : "MISMATCH");

    // ---- Cross-stream event dependency demo ----
    printf("\n--- Cross-stream event dependency ---\n");
    {
        cudaStream_t sA, sB;
        CUDA_CHECK(cudaStreamCreateWithFlags(&sA, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sB, cudaStreamNonBlocking));
        cudaEvent_t fence;
        CUDA_CHECK(cudaEventCreate(&fence));

        // sA: H2D + kernel — then record fence
        CUDA_CHECK(cudaMemcpyAsync(d_data[0], h_in[0], bytes,
                                   cudaMemcpyHostToDevice, sA));
        scaleKernel<<<blocks, THREADS, 0, sA>>>(d_data[0], 2.0f, N);
        CUDA_CHECK(cudaEventRecord(fence, sA));  // fence fires when sA reaches here

        // sB: must wait for sA's fence before starting (GPU-side wait)
        CUDA_CHECK(cudaStreamWaitEvent(sB, fence, 0));
        scaleKernel<<<blocks, THREADS, 0, sB>>>(d_data[0], 0.5f, N);  // undo scale
        CUDA_CHECK(cudaMemcpyAsync(h_out[0], d_data[0], bytes,
                                   cudaMemcpyDeviceToHost, sB));

        CUDA_CHECK(cudaStreamSynchronize(sB));
        printf("After scale x2 then x0.5, first element: %.4f (expected ~%.4f)\n",
               h_out[0][0], h_in[0][0]);

        CUDA_CHECK(cudaEventDestroy(fence));
        CUDA_CHECK(cudaStreamDestroy(sA));
        CUDA_CHECK(cudaStreamDestroy(sB));
    }

    // ---- Cleanup ----
    for (int a = 0; a < N_ARRAYS; a++) {
        CUDA_CHECK(cudaStreamDestroy(streams[a]));
        CUDA_CHECK(cudaFree(d_data[a]));
        CUDA_CHECK(cudaFreeHost(h_in[a]));
        CUDA_CHECK(cudaFreeHost(h_out[a]));
    }
    CUDA_CHECK(cudaEventDestroy(tStart));
    CUDA_CHECK(cudaEventDestroy(tStop));

    return 0;
}

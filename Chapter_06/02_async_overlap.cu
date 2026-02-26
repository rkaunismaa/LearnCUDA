/*
 * Chapter 06 — 02_async_overlap.cu
 *
 * Double-buffered pipeline: overlaps H2D copy, kernel compute, and D2H copy
 * for large arrays split into chunks.
 *
 * Compares:
 *   A) Synchronous: H2D → Kernel → D2H sequentially for each chunk
 *   B) Pipelined: use 2 streams so chunk N+1's H2D overlaps chunk N's kernel
 *
 * Timeline goal:
 *   Stream 0: [H2D 0] [Kernel 0]         [D2H 0]
 *   Stream 1:         [H2D 1]  [Kernel 1]         [D2H 1]
 *   Stream 0:                  [H2D 2]   [Kernel 2]         [D2H 2]
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o async_overlap 02_async_overlap.cu
 * Run:
 *   ./async_overlap
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

// Kernel: heavy-ish compute to make kernel time comparable to transfer time
__global__ void processChunk(float *data, int n, int iter)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = data[i];
        for (int k = 0; k < iter; k++)
            v = sinf(v) * 0.9f + 0.1f;
        data[i] = v;
    }
}

int main()
{
    const int TOTAL   = 1 << 23;  // 8M floats = 32 MB
    const int CHUNKS  = 8;        // Split into 8 chunks
    const int CHUNK_N = TOTAL / CHUNKS;
    const int ITERS   = 30;       // Kernel iterations (tune to balance with transfer)
    const int THREADS = 256;

    size_t totalBytes = (size_t)TOTAL   * sizeof(float);
    size_t chunkBytes = (size_t)CHUNK_N * sizeof(float);

    printf("Async Pipeline Demo\n");
    printf("Total: %d floats (%.1f MB), %d chunks of %.1f MB each\n\n",
           TOTAL, totalBytes / 1e6f, CHUNKS, chunkBytes / 1e6f);

    // ---- Allocate PINNED host buffers ----
    float *h_in, *h_out;
    CUDA_CHECK(cudaMallocHost(&h_in,  totalBytes));
    CUDA_CHECK(cudaMallocHost(&h_out, totalBytes));
    for (int i = 0; i < TOTAL; i++) h_in[i] = (float)i / TOTAL;

    // ---- Allocate device buffers: 2 ping-pong buffers ----
    float *d_buf[2];
    CUDA_CHECK(cudaMalloc(&d_buf[0], chunkBytes));
    CUDA_CHECK(cudaMalloc(&d_buf[1], chunkBytes));

    int blocks = (CHUNK_N + THREADS - 1) / THREADS;

    cudaEvent_t tStart, tStop;
    CUDA_CHECK(cudaEventCreate(&tStart));
    CUDA_CHECK(cudaEventCreate(&tStop));

    // ================================================================
    // A: SYNCHRONOUS (no overlap)
    //    For each chunk: H2D → kernel → D2H, one at a time
    // ================================================================
    CUDA_CHECK(cudaEventRecord(tStart));
    for (int c = 0; c < CHUNKS; c++) {
        float *h_src = h_in  + c * CHUNK_N;
        float *h_dst = h_out + c * CHUNK_N;

        cudaMemcpy(d_buf[0], h_src, chunkBytes, cudaMemcpyHostToDevice);
        processChunk<<<blocks, THREADS>>>(d_buf[0], CHUNK_N, ITERS);
        cudaMemcpy(h_dst, d_buf[0], chunkBytes, cudaMemcpyDeviceToHost);
    }
    CUDA_CHECK(cudaEventRecord(tStop));
    CUDA_CHECK(cudaEventSynchronize(tStop));
    float ms_sync;
    CUDA_CHECK(cudaEventElapsedTime(&ms_sync, tStart, tStop));
    printf("Synchronous (no overlap):   %.3f ms\n", ms_sync);

    // ================================================================
    // B: PIPELINED (double-buffered with 2 streams)
    //    Chunk c uses stream c%2 and buffer d_buf[c%2]
    //    While chunk c's kernel runs on stream 0,
    //    chunk c+1's H2D runs on stream 1 (different DMA engine)
    // ================================================================
    cudaStream_t streams[2];
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking));

    // Reset output for verification
    for (int i = 0; i < TOTAL; i++) h_out[i] = 0.0f;

    CUDA_CHECK(cudaEventRecord(tStart));
    for (int c = 0; c < CHUNKS; c++) {
        int   s    = c % 2;             // Alternate between stream 0 and 1
        float *src = h_in  + c * CHUNK_N;
        float *dst = h_out + c * CHUNK_N;

        // Async H2D for this chunk (may run while previous kernel is executing)
        CUDA_CHECK(cudaMemcpyAsync(d_buf[s], src, chunkBytes,
                                   cudaMemcpyHostToDevice, streams[s]));

        // Kernel for this chunk (follows H2D on same stream — guaranteed ordering)
        processChunk<<<blocks, THREADS, 0, streams[s]>>>(
            d_buf[s], CHUNK_N, ITERS);

        // Async D2H for this chunk (follows kernel on same stream)
        CUDA_CHECK(cudaMemcpyAsync(dst, d_buf[s], chunkBytes,
                                   cudaMemcpyDeviceToHost, streams[s]));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(tStop));
    CUDA_CHECK(cudaEventSynchronize(tStop));
    float ms_pipe;
    CUDA_CHECK(cudaEventElapsedTime(&ms_pipe, tStart, tStop));
    printf("Pipelined (2 streams):      %.3f ms\n", ms_pipe);
    printf("Speedup:                    %.2fx\n\n", ms_sync / ms_pipe);

    // ---- Verify pipelined result matches synchronous ----
    // Re-run synchronous to get reference output in h_out reference
    float *h_ref;
    CUDA_CHECK(cudaMallocHost(&h_ref, totalBytes));
    for (int c = 0; c < CHUNKS; c++) {
        cudaMemcpy(d_buf[0], h_in + c * CHUNK_N, chunkBytes, cudaMemcpyHostToDevice);
        processChunk<<<blocks, THREADS>>>(d_buf[0], CHUNK_N, ITERS);
        cudaMemcpy(h_ref + c * CHUNK_N, d_buf[0], chunkBytes, cudaMemcpyDeviceToHost);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    int errors = 0;
    for (int i = 0; i < TOTAL; i++) {
        if (fabsf(h_out[i] - h_ref[i]) > 1e-4f) {
            if (errors < 3) printf("  Mismatch at %d: %.6f vs %.6f\n",
                                   i, h_out[i], h_ref[i]);
            errors++;
        }
    }
    printf("Verification: %s (%d errors)\n", errors ? "FAILED" : "PASSED", errors);

    printf("\nNote: Actual overlap and speedup depends on the balance between\n");
    printf("transfer time and kernel time. Use Nsight Systems to visualize.\n");
    printf("Command: nsys profile --trace=cuda ./async_overlap\n");

    // ---- Cleanup ----
    CUDA_CHECK(cudaStreamDestroy(streams[0]));
    CUDA_CHECK(cudaStreamDestroy(streams[1]));
    CUDA_CHECK(cudaFree(d_buf[0]));
    CUDA_CHECK(cudaFree(d_buf[1]));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaFreeHost(h_ref));
    CUDA_CHECK(cudaEventDestroy(tStart));
    CUDA_CHECK(cudaEventDestroy(tStop));

    return 0;
}

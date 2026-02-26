/*
 * Chapter 09 — 03_cooperative_groups.cu
 *
 * Demonstrates the Cooperative Groups API (CUDA 9+):
 *   - cg::this_thread_block() for block-level groups
 *   - cg::tiled_partition<N>() for sub-block groups
 *   - cg::reduce() for built-in reduction
 *   - Group shuffle and sync methods
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o coop_groups 03_cooperative_groups.cu
 * Run:
 *   ./coop_groups
 */

#include <stdio.h>
#include <stdlib.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

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
// Generic reduce using cooperative groups — works for any tile size
// This is the "modern CUDA" way to write reduction code
// ================================================================
template<typename Group>
__device__ float groupReduceSum(Group g, float val)
{
    // cg::reduce performs a hardware-accelerated reduction within the group
    return cg::reduce(g, val, cg::plus<float>());
}

// ================================================================
// Kernel: demonstrates different group sizes
// ================================================================
__global__ void cooperativeGroupsDemo(const float *in, float *out_warp,
                                       float *out_halfwarp, int n)
{
    cg::thread_block block = cg::this_thread_block();

    // Warp-level group (32 threads)
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Half-warp group (16 threads per group, 2 per warp)
    cg::thread_block_tile<16> half_warp = cg::tiled_partition<16>(block);

    // Quarter-warp (8 threads per group)
    cg::thread_block_tile<8> quarter_warp = cg::tiled_partition<8>(block);

    int i = block.group_index().x * block.group_dim().x + block.thread_index().x;
    float val = (i < n) ? in[i] : 0.0f;

    // Reduce within each warp (32 threads) — result in lane 0 of each warp
    float warp_sum = groupReduceSum(warp, val);

    // Reduce within each half-warp (16 threads) — result in thread 0 of each group
    float hw_sum = groupReduceSum(half_warp, val);

    // Store warp sums (one per warp)
    if (warp.thread_rank() == 0) {
        int warp_id = i / 32;
        if (warp_id < (n + 31) / 32)
            out_warp[warp_id] = warp_sum;
    }

    // Store half-warp sums (one per half-warp)
    if (half_warp.thread_rank() == 0) {
        int hw_id = i / 16;
        if (hw_id < (n + 15) / 16)
            out_halfwarp[hw_id] = hw_sum;
    }
}

// ================================================================
// Group-based block reduction: demonstrates composable reduction
// ================================================================
__global__ void blockReduceDemo(const float *in, float *out, int n)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    __shared__ float warp_sums[32];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < n) ? in[i] : 0.0f;

    // Warp reduce
    float wsum = cg::reduce(warp, val, cg::plus<float>());

    // First thread of each warp stores to shared memory
    if (warp.thread_rank() == 0)
        warp_sums[warp.meta_group_rank()] = wsum;

    block.sync();  // Cooperative groups-style sync (equivalent to __syncthreads)

    // First warp reduces warp sums
    int nwarps = blockDim.x / 32;
    float block_sum = 0.0f;
    if (threadIdx.x < nwarps) {
        cg::thread_block_tile<32> final_warp = cg::tiled_partition<32>(block);
        block_sum = cg::reduce(final_warp,
                               (threadIdx.x < nwarps) ? warp_sums[threadIdx.x] : 0.0f,
                               cg::plus<float>());
    }

    if (threadIdx.x == 0)
        out[blockIdx.x] = block_sum;
}

int main()
{
    const int N = 1 << 10;  // Small for easy verification
    int THREADS = 128, BLOCKS = (N + THREADS - 1) / THREADS;

    float *h_in = (float*)malloc(N * sizeof(float));
    double ref_sum = 0.0;
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)(i % 32 + 1);  // 1..32 cycling
        ref_sum += h_in[i];
    }

    float *d_in, *d_warp, *d_hw;
    int n_warps = (N + 31) / 32;
    int n_hw    = (N + 15) / 16;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_warp, n_warps * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hw,   n_hw * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // ---- Cooperative Groups Demo ----
    printf("=== Cooperative Groups Demo ===\n");
    cooperativeGroupsDemo<<<BLOCKS, THREADS>>>(d_in, d_warp, d_hw, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *h_warp = (float*)malloc(n_warps * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_warp, d_warp, n_warps * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Warp sums (first 4 warps of 32 threads each, values 1..32):\n");
    for (int w = 0; w < 4 && w < n_warps; w++)
        printf("  Warp %d sum: %.0f (expected %.0f)\n",
               w, h_warp[w], (float)(32 * 33 / 2));  // sum of 1..32 = 528

    // ---- Block reduce ----
    printf("\n=== Block Reduce (cg::reduce) ===\n");
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, BLOCKS * sizeof(float)));
    blockReduceDemo<<<BLOCKS, THREADS>>>(d_in, d_out, N);

    // Final reduce on CPU
    float *h_out = (float*)malloc(BLOCKS * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, BLOCKS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    double gpu_sum = 0.0;
    for (int b = 0; b < BLOCKS; b++) gpu_sum += h_out[b];
    printf("Reference sum: %.1f\n", ref_sum);
    printf("GPU sum (via cg::reduce): %.1f\n", gpu_sum);
    printf("Match: %s\n", fabsf((float)(gpu_sum - ref_sum)) < 1.0f ? "YES" : "NO");

    printf("\nCooperative Groups advantages:\n");
    printf("  - Tile size is a template parameter → compiler can optimize\n");
    printf("  - group.sync() / group.shfl_down() are portable\n");
    printf("  - Works for 2, 4, 8, 16, 32 thread tiles\n");

    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_warp));
    CUDA_CHECK(cudaFree(d_hw)); CUDA_CHECK(cudaFree(d_out));
    free(h_in); free(h_warp); free(h_out);
    return 0;
}

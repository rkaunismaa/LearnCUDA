# Chapter 09: Warp-Level Primitives

## 9.1 The Warp as a Programming Unit

So far we've treated threads as independent workers that communicate only through shared memory. But the GPU hardware executes threads in **warps of 32** that are intrinsically synchronized. We can exploit this for faster communication without shared memory.

## 9.2 Warp Divergence

All 32 threads in a warp execute the same instruction. When different threads need to take different branches, the warp must execute both paths — threads that don't take the current path are **masked off** (idle):

```c
__global__ void divergent(float *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i % 2 == 0) {          // Threads 0,2,4,... take this path
        data[i] = data[i] * 2; // Threads 1,3,5,... are IDLE here
    } else {
        data[i] = data[i] + 1; // Threads 0,2,4,... are IDLE here
    }
    // Effective throughput: 50% for both branches
}
```

If all threads in a warp take the same branch, there is **no** divergence cost. Divergence is only a problem when threads in the **same warp** diverge.

### Measuring Divergence: `__ballot_sync`

```c
// Returns a 32-bit mask where bit i is set if thread i's condition is true
unsigned mask = 0xffffffff;
unsigned ballot = __ballot_sync(mask, condition);
int active_count = __popc(ballot);  // Count active threads
```

## 9.3 Warp Shuffle Instructions

Shuffle instructions let threads in a warp **exchange registers directly** — no shared memory needed, no `__syncthreads()` needed:

```c
unsigned mask = 0xffffffff;

// Thread i gets value from thread srcLane
float val = __shfl_sync(mask, val, srcLane);

// Thread i gets value from thread (i + delta) — wraps within warp
float val = __shfl_down_sync(mask, val, delta);

// Thread i gets value from thread (i - delta)
float val = __shfl_up_sync(mask, val, delta);

// Thread i gets value from thread (i XOR mask) — butterfly pattern
float val = __shfl_xor_sync(mask, val, laneMask);
```

The first argument is the **active thread mask** — `0xffffffff` means all 32 lanes participate.

### Warp Reduction Without Shared Memory

```c
__device__ float warpSum(float val)
{
    unsigned mask = 0xffffffff;
    val += __shfl_down_sync(mask, val, 16);  // Sum with thread i+16
    val += __shfl_down_sync(mask, val,  8);  // Sum with thread i+8
    val += __shfl_down_sync(mask, val,  4);
    val += __shfl_down_sync(mask, val,  2);
    val += __shfl_down_sync(mask, val,  1);
    return val;  // Thread 0 has the warp sum
}
```

### Warp Broadcast

```c
// Broadcast lane 0's value to all threads in the warp
float val = __shfl_sync(0xffffffff, my_val, 0);
```

## 9.4 Warp Voting

```c
unsigned mask = 0xffffffff;

// true if ANY thread has condition=true
bool any = __any_sync(mask, condition);

// true if ALL threads have condition=true
bool all = __all_sync(mask, condition);

// 32-bit mask of which threads have condition=true
unsigned ballot = __ballot_sync(mask, condition);
```

Useful for early exit, conditional work, and debugging.

## 9.5 Cooperative Groups

Cooperative Groups (CUDA 9+) provide a safe, flexible API for synchronizing groups of threads at different granularities:

```c
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void myKernel(float *data)
{
    // Block-level group (equivalent to __syncthreads())
    cg::thread_block block = cg::this_thread_block();

    // Warp-level group (32 threads)
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Sub-warp group (16 threads — first or second half of warp)
    cg::thread_block_tile<16> half_warp = cg::tiled_partition<16>(block);

    // Sync the block
    block.sync();

    // Sync just the warp — faster than block.sync()
    warp.sync();

    // Warp shuffle via cooperative groups (same as __shfl_down_sync)
    float val = data[block.thread_rank()];
    val += warp.shfl_down(val, 16);
    // ...

    // Built-in warp reduce (CUDA 11+)
    float sum = cg::reduce(warp, val, cg::plus<float>());
}
```

Cooperative groups make warp-level code more readable and portable.

## 9.6 Exercises

1. In `01_warp_divergence.cu`, restructure the divergent kernel to use `__ballot_sync` to determine which threads need extra work, then apply it only to those threads. Does it help?
2. Implement a warp-level prefix scan (inclusive sum) using `__shfl_up_sync`.
3. In `03_cooperative_groups.cu`, use `cg::tiled_partition<8>` to create sub-warp groups of 8 threads. Implement a reduction within each group.
4. Implement a warp-based bitonic sort for 32 elements using `__shfl_xor_sync`.

## 9.7 Key Takeaways

- Warp divergence only hurts performance when threads in the **same warp** diverge.
- `__shfl_down_sync` enables warp-level reduction with **no shared memory and no __syncthreads()**.
- Always pass an explicit active thread mask (`0xffffffff` for full warps) to shuffle/vote functions.
- Cooperative groups provide a cleaner API for warp-level programming.
- `cg::reduce(warp, val, cg::plus<float>())` is the simplest way to do a warp reduction in modern CUDA.

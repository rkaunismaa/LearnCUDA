# Chapter 05: Parallel Reduction and Atomic Operations

## 5.1 What is a Reduction?

A **reduction** collapses an array of N values into a single value using an associative operator (sum, max, min, product, etc.):

```
Input:  [3, 1, 4, 1, 5, 9, 2, 6]
Output: 31  (sum)
```

On the CPU this is trivial — a single loop. On the GPU, doing it with one thread wastes all the parallelism. We need a parallel algorithm.

## 5.2 The Parallel Reduction Tree

The key insight: split the work recursively. At each step, half the threads compute pairwise sums. After log₂(N) steps, a single value remains.

```
Step 0 (8 values):  [3][1][4][1][5][9][2][6]
                     ↕   ↕   ↕   ↕
Step 1 (4 values):  [4] [5] [14][8]
                     ↕       ↕
Step 2 (2 values):  [9]     [22]
                     ↕
Step 3 (1 value):  [31]
```

This runs in O(log₂N) steps instead of O(N) — but uses O(N) threads total.

## 5.3 Naive Reduction: Interleaved Addressing (Divergent)

```c
// BAD: interleaved stride causes warp divergence
__global__ void reduceInterleaved(float *data, int n)
{
    int tid = threadIdx.x;
    // Stride doubles each step
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0)       // ← divergence! half threads idle
            data[tid] += data[tid + stride];
        __syncthreads();
    }
}
```

**Problem**: `tid % (2 * stride) == 0` means threads 0, 2, 4, 8, ... are active. Threads 0 and 1 are in the same warp. Thread 1 is idle while thread 0 works — **warp divergence** wastes 50% of warp capacity.

## 5.4 Better: Sequential Addressing (No Divergence)

```c
// GOOD: sequential addressing — no warp divergence
__global__ void reduceSequential(float *data, int n)
{
    int tid = threadIdx.x;
    // Stride halves each step, threads at the low end stay active
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)                  // ← contiguous threads active
            data[tid] += data[tid + stride];
        __syncthreads();
    }
}
```

Now threads 0..stride-1 are always active — they form **contiguous warps** with no divergence. This alone doubles performance.

## 5.5 Warp-Level Reduction with Shuffle Instructions

For the final 32 elements (one warp), we can skip shared memory entirely using `__shfl_down_sync`:

```c
// Warp-level reduce: all 32 threads cooperate, result in lane 0
__device__ float warpReduceSum(float val)
{
    unsigned mask = 0xffffffff;  // All 32 threads participate
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);
    return val;  // Only lane 0 has the final sum
}
```

`__shfl_down_sync(mask, val, offset)`: thread `i` receives the value from thread `i + offset`, no shared memory required.

```
Before: lane  0  1  2  3  4  5  6  7  (only showing 8 lanes)
val:         [3][1][4][1][5][9][2][6]

offset=4:  lane i += lane i+4
val:         [8][10][6][7][5][9][2][6]

offset=2:  lane i += lane i+2
val:         [14][19][6][7][5][9][2][6]

offset=1:  lane i += lane i+1
val:         [33][19]...

Lane 0 = 33 (correct sum)
```

## 5.6 Full Optimized Reduction

Combining per-block warp-level reduction with a two-pass approach:

```
Pass 1: Each block reduces its portion → one partial sum per block
Pass 2: A second kernel reduces the partial sums → final answer
```

See `02_reduction_optimized.cu` for the full implementation.

## 5.7 Atomic Operations

**Atomics** allow multiple threads to safely update the same memory location without race conditions:

```c
// Without atomic: RACE CONDITION (threads may overwrite each other)
*counter = *counter + 1;  // Read-modify-write is not atomic!

// With atomic: SAFE
atomicAdd(counter, 1);    // Hardware guarantees this is indivisible
```

### Available Atomic Operations

| Operation | Function | Supported Types |
|-----------|----------|----------------|
| Add | `atomicAdd(addr, val)` | int, float, double |
| Subtract | `atomicSub(addr, val)` | int, unsigned |
| Exchange | `atomicExch(addr, val)` | int, float |
| Min | `atomicMin(addr, val)` | int, unsigned |
| Max | `atomicMax(addr, val)` | int, unsigned |
| AND | `atomicAnd(addr, val)` | int, unsigned |
| OR | `atomicOr(addr, val)` | int, unsigned |
| Compare-and-Swap | `atomicCAS(addr, cmp, val)` | int, unsigned, 64-bit |

### The atomicCAS Pattern

`atomicCAS(addr, compare, val)` atomically: if `*addr == compare`, set `*addr = val`, return old value. This is a universal building block — you can implement any atomic operation with it:

```c
// Float atomicMin using CAS (not natively provided)
__device__ float atomicMinFloat(float *addr, float val)
{
    int *addr_i = (int*)addr;
    int old = *addr_i, expected;
    do {
        expected = old;
        old = atomicCAS(addr_i, expected,
                        __float_as_int(fminf(val, __int_as_float(expected))));
    } while (old != expected);
    return __int_as_float(old);
}
```

### Atomic Contention

Atomics are **serialized** when multiple threads hit the same address — exactly what happens in a histogram where many inputs map to few bins:

```c
// BAD: all threads fight over 256 bins — massive contention
atomicAdd(&global_hist[bin[i]], 1);

// BETTER: privatized per-block histogram in shared memory
__shared__ int smem_hist[256];
atomicAdd(&smem_hist[local_bin], 1);  // low contention, fast shared mem
__syncthreads();
// Then merge smem_hist → global_hist once per block
atomicAdd(&global_hist[j], smem_hist[j]);
```

See `03_atomics.cu` for the full histogram example.

## 5.8 Exercises

1. Run `01_reduction_naive.cu`. What is the speedup of sequential over interleaved addressing?
2. Modify the warp shuffle reduction to compute the **maximum** instead of sum.
3. In `03_atomics.cu`, increase the number of threads per bin to create more contention. At what point does the privatized histogram approach break even?
4. Implement a parallel prefix sum (scan) using warp shuffle `__shfl_up_sync`.
5. What is the theoretical minimum number of steps for reducing 1024 elements in a single block?

## 5.9 Key Takeaways

- Use **sequential addressing** (not interleaved) to avoid warp divergence.
- **Warp shuffle** (`__shfl_down_sync`) eliminates shared memory for intra-warp reduction.
- Full block reduction: load to registers → warp reduce → write to shared memory → warp reduce → block sum.
- Two-pass reduction: block partials in pass 1, reduce partials in pass 2.
- **Atomics are serialized** at the same address — use private shared memory histograms for histogram-like workloads.
- `atomicCAS` is the universal building block for custom atomic operations.

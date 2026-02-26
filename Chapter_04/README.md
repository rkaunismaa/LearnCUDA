# Chapter 04: Tiled Matrix Multiplication — Shared Memory in Practice

## 4.1 The Classic GPU Benchmark

Matrix multiplication (`C = A × B`) is the most important kernel in GPU computing. It is:
- The backbone of deep learning (every linear layer = GEMM)
- The canonical example for teaching shared memory optimization
- Used to benchmark GPU performance (GFLOPS/s)

For matrices A (M×K) and B (K×N), each element of C is a dot product:
```
C[row][col] = Σ A[row][k] * B[k][col]  for k = 0..K-1
```

Total operations: **M × N × 2K** (K multiplications + K additions per output).

## 4.2 Naive Matrix Multiplication

The straightforward approach: one thread computes one element of C.

```c
__global__ void matmulNaive(float *A, float *B, float *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];  // ← problem here!
        C[row * N + col] = sum;
    }
}
```

**Problem with B access**: Each thread in a column reads the same row of A (coalesced) but reads **different rows of B** — `B[k][col]` means column-major access, which is strided and non-coalesced.

```
Thread 0 reads: A[row][0..K-1]  ← coalesced (consecutive in memory)
                B[0..K-1][0]    ← NON-coalesced (K rows apart!)

Thread 1 reads: A[row][0..K-1]  ← coalesced
                B[0..K-1][1]    ← NON-coalesced
```

All threads in a warp access the same column of B but different rows → worst case memory access pattern!

## 4.3 Tiled Matrix Multiplication

The solution: load tiles of A and B into shared memory, then compute from fast on-chip memory.

### The Tiling Idea

Instead of computing the full dot product in one pass, we break it into **tiles of width TILE**:

```
C[row][col] = Σ_{tile=0}^{K/TILE-1}
              A_tile[row][0..TILE-1] · B_tile[0..TILE-1][col]
```

For each tile:
1. **Cooperatively load** a TILE×TILE block of A and B into shared memory
2. **Compute** partial dot products from shared memory
3. **Accumulate** into the running sum

### Memory Access Pattern with Tiling

```
Tile iteration 0:                    Tile iteration 1:
┌──────────┐                         ┌──────────┐
│  A_tile  │ ← loaded cooperatively  │  A_tile  │
└──────────┘                         └──────────┘
      ×                                    ×
┌──────────┐                         ┌──────────┐
│  B_tile  │ ← loaded cooperatively  │  B_tile  │
└──────────┘                         └──────────┘
```

Key insight: **memory traffic is reduced by a factor of TILE**.
- Without tiling: each thread loads K elements of A and K elements of B
- With tiling: the block of TILE² threads collectively loads TILE² elements of A and TILE² elements of B, and each is reused TILE times

**Arithmetic intensity** increases from 0.25 FLOP/byte (naive) to TILE/8 FLOP/byte (tiled). For TILE=16, that's 2 FLOP/byte — approaching practical GPU limits!

## 4.4 The Tiled Algorithm Step-by-Step

```
For each output tile [row_block, col_block]:
  acc = 0.0

  For each K-tile (tile_k = 0, TILE, 2*TILE, ...):
    // Phase 1: Collaborative loading into shared memory
    thread (ty, tx) loads:
      A_smem[ty][tx] = A[row][tile_k + tx]
      B_smem[ty][tx] = B[tile_k + ty][col]

    __syncthreads()  // Wait for all loads to complete

    // Phase 2: Compute partial dot product from shared memory
    For k = 0..TILE-1:
      acc += A_smem[ty][k] * B_smem[k][tx]

    __syncthreads()  // Wait for all computes before next tile's loads

  C[row][col] = acc
```

**Two `__syncthreads()` calls per tile iteration** are required:
1. After loading: prevent reading before all threads finish loading
2. After computing: prevent overwriting shared memory before all threads finish computing

## 4.5 Benchmarking Matrix Multiply

For a 2048×2048 float32 matrix multiplication:
- Total ops: 2048³ × 2 ≈ 17.2 GFLOP
- Naive: ~5-20 GFLOPS (limited by memory bandwidth)
- Tiled (TILE=16): ~200-500 GFLOPS
- cuBLAS: ~18-22 TFLOPS (near peak for this problem size!)

The gap between tiled and cuBLAS represents further optimizations we'll see in later chapters: register tiling, vectorized loads, software pipelining, warp specialization.

## 4.6 Beyond Basic Tiling

Modern CUDA matrix multiplication implementations use many additional tricks:
- **Double buffering**: overlap loading tile N+1 while computing tile N (Chapter 06)
- **Register blocking**: each thread computes a small sub-tile (e.g., 8×8 output elements) from registers
- **Tensor Cores**: hardware units for 4×4 or larger matrix ops (Chapter 12)
- **Vectorized loads**: use `float4` to load 16 bytes at once

See `01_matmul_naive.cu` and `02_matmul_tiled.cu`.

## 4.7 Exercises

1. Run the naive vs tiled benchmarks. Calculate the achieved GFLOPS for each.
2. Experiment with different `TILE` sizes (8, 16, 32). How does performance change? Why might 32 be slower than 16 despite more reuse?
3. What happens if K is not a multiple of TILE? Look at how boundary conditions are handled in the code.
4. The tiled kernel still has non-coalesced access for one of the matrices. Which one? How would you fix it? (Hint: think about how B_smem is loaded.)
5. Implement a version that verifies against a CPU reference for a small matrix (e.g., 64×64).

## 4.8 Key Takeaways

- Matrix multiply is the canonical shared memory optimization problem.
- **Tiling** reduces global memory traffic by factor TILE (dramatically improves arithmetic intensity).
- Both `__syncthreads()` calls (after load and after compute) are required.
- A tile size of 16 is common (16×16 = 256 threads, fits within shared memory limits).
- Even optimized tiled code is still far below cuBLAS — many more tricks exist.

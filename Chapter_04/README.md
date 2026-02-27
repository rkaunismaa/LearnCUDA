# Chapter 04: Tiled Matrix Multiplication — Shared Memory in Practice

## 4.1 The Classic GPU Benchmark

Matrix multiplication (`C = A × B`) is the most important kernel in GPU computing. It is:
- The backbone of deep learning (every linear layer = GEMM)
- The canonical example for teaching shared memory optimization
- Used to benchmark GPU performance (GFLOPS/s)

```mermaid
graph LR
    subgraph A["Matrix A  (M rows × K cols)"]
        AROW["Row r:\n[ a₀  a₁  a₂  ···  aK₋₁ ]"]
    end
    subgraph B["Matrix B  (K rows × N cols)"]
        BCOL["Col c:\n[ b₀  b₁  b₂  ···  bK₋₁ ]ᵀ"]
    end
    subgraph C["Matrix C  (M rows × N cols)"]
        CE["C[r][c] = Σₖ A[r][k] × B[k][c]\nOne thread computes this dot product\n2K operations per element"]
    end
    AROW -->|"dot product"| CE
    BCOL -->|"dot product"| CE

    style A  fill:#1f618d,color:#fff,stroke:#154360
    style B  fill:#1e8449,color:#fff,stroke:#196f3d
    style C  fill:#7d3c98,color:#fff,stroke:#6c3483
    style AROW fill:#154360,color:#aed6f1,stroke:#1f618d
    style BCOL fill:#196f3d,color:#a9dfbf,stroke:#1e8449
    style CE   fill:#6c3483,color:#d7bde2,stroke:#7d3c98
```

For matrices A (M×K) and B (K×N), each element of C is a dot product:
```
C[row][col] = Σ A[row][k] * B[k][col]  for k = 0..K-1
```

Total operations: **M × N × 2K** (K multiplications + K additions per output element).

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
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}
```

### The Data Reuse Problem

The naive kernel loads every value from global memory independently — zero cooperation between threads, zero reuse:

```diff
  Naive kernel — warp of 32 threads computing C[row][0..31]
  Each thread independently loads K values of A and K values of B

  Thread 0 computes C[row][0]:
- Loads A[row][0], A[row][1], ..., A[row][K-1]  from global mem  (K reads)
- Loads B[0][0],   B[1][0],   ..., B[K-1][0]    from global mem  (K reads)

  Thread 1 computes C[row][1]:
- Loads A[row][0], A[row][1], ..., A[row][K-1]  from global mem  (K reads) ← SAME as Thread 0!
- Loads B[0][1],   B[1][1],   ..., B[K-1][1]    from global mem  (K reads)

  Thread 2 computes C[row][2]:
- Loads A[row][0], A[row][1], ..., A[row][K-1]  from global mem  (K reads) ← SAME again!
- Loads B[0][2],   B[1][2],   ..., B[K-1][2]    from global mem  (K reads)
  ...
  Thread 31 computes C[row][31]:
- Loads A[row][0..K-1] from global mem  ← 32nd redundant copy of row r of A ✗
- Loads B[0..K-1][31]  from global mem

  Total global reads for this warp: 32 × 2K = 64K  (should be: 32K + K = 33K)
  Wasted bandwidth: ~49%  |  Arithmetic intensity: ~0.25 FLOP/byte ✗
```

## 4.3 Tiled Matrix Multiplication

The solution: load tiles of A and B into shared memory, then compute from fast on-chip memory. All threads in the block **share** the loaded tile — K reads become K/TILE reads.

### The Tiling Idea

Instead of computing the full dot product in one pass, break the K dimension into **tiles of width TILE**:

```mermaid
graph LR
    subgraph K["K dimension split into tiles of width TILE"]
        T0["Tile 0\nk = 0..T-1"]
        T1["Tile 1\nk = T..2T-1"]
        T2["Tile 2\nk = 2T..3T-1"]
        TN["Tile K/T-1\nk = K-T..K-1"]
    end

    subgraph SMEM0["Shared Memory — Tile 0"]
        AS0["A_smem\nT×T floats"]
        BS0["B_smem\nT×T floats"]
    end

    subgraph SMEM1["Shared Memory — Tile 1"]
        AS1["A_smem\nT×T floats"]
        BS1["B_smem\nT×T floats"]
    end

    T0 -->|"load into"| SMEM0
    SMEM0 -->|"compute partial C\nacc += A_smem × B_smem"| T1
    T1 -->|"load into"| SMEM1
    SMEM1 -->|"compute partial C\nacc += ..."| T2

    style K     fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style T0    fill:#1f618d,color:#fff,stroke:#154360
    style T1    fill:#1f618d,color:#fff,stroke:#154360
    style T2    fill:#1f618d,color:#fff,stroke:#154360
    style TN    fill:#1f618d,color:#fff,stroke:#154360
    style SMEM0 fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style SMEM1 fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style AS0   fill:#d35400,color:#fff,stroke:#a04000
    style BS0   fill:#1e8449,color:#fff,stroke:#196f3d
    style AS1   fill:#d35400,color:#fff,stroke:#a04000
    style BS1   fill:#1e8449,color:#fff,stroke:#196f3d
```

Key insight: **memory traffic is reduced by a factor of TILE**.
- Without tiling: each thread loads K elements of A and K elements of B independently
- With tiling: the block of TILE² threads loads each tile **once** and every element is reused TILE times

### Cooperative Tile Loading

```mermaid
graph TB
    subgraph GLOBAL["💾 Global Memory  (~600 cycles latency)"]
        GA["A[ row_start : row_start+T ][ tile*T : (tile+1)*T ]\nT×T floats — each read once per tile iteration"]
        GB["B[ tile*T : (tile+1)*T ][ col_start : col_start+T ]\nT×T floats — each read once per tile iteration"]
    end

    subgraph COOP["🤝 Cooperative Load  (all T² threads work together)"]
        RULE["Thread (ty, tx) is responsible for:\nA_smem[ty][tx] = A[row_start+ty][tile*T+tx]\nB_smem[ty][tx] = B[tile*T+ty][col_start+tx]\nEach thread loads exactly 1 element of each tile"]
    end

    subgraph SMEM["🔥 Shared Memory  (~5–30 cycles latency)"]
        AS["A_smem[TILE][TILE]\nEvery thread in the block\ncan read any element ✓"]
        BS["B_smem[TILE][TILE]\nEvery thread in the block\ncan read any element ✓"]
    end

    GA -->|"T² parallel loads"| COOP
    GB -->|"T² parallel loads"| COOP
    COOP --> AS
    COOP --> BS

    style GLOBAL fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style COOP   fill:#2c3e50,color:#ecf0f1,stroke:#1a252f
    style SMEM   fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style GA     fill:#1f618d,color:#fff,stroke:#154360
    style GB     fill:#1e8449,color:#fff,stroke:#196f3d
    style RULE   fill:#1a252f,color:#ecf0f1,stroke:#2c3e50
    style AS     fill:#d35400,color:#fff,stroke:#a04000
    style BS     fill:#1e8449,color:#fff,stroke:#196f3d
```

## 4.4 The Tiled Algorithm Step-by-Step

```mermaid
flowchart TD
    INIT["🔢 Initialize\nacc = 0.0f\n(one accumulator per output element)"]

    TILELOOP{"Next tile?\ntile = 0, 1, ..., K/T-1"}

    LOAD["📥 Phase 1 — Cooperative Load\nAll T² threads load in parallel:\nA_smem[ty][tx] = A[row_start + ty][tile*T + tx]\nB_smem[ty][tx] = B[tile*T + ty][col_start + tx]"]

    SYNC1["⬛ __syncthreads()  ①\nBarrier — wait until ALL threads\nfinish loading before any thread reads"]

    COMPUTE["⚡ Phase 2 — Compute Partial Dot Product\nfor k = 0 .. T-1:\n    acc += A_smem[ty][k] * B_smem[k][tx]\n(all arithmetic from fast shared memory)"]

    SYNC2["⬛ __syncthreads()  ②\nBarrier — wait until ALL threads\nfinish computing before next tile\noverwrites shared memory"]

    WRITE["📤 Write Result\nC[row][col] = acc\n(one global write per thread)"]

    INIT --> TILELOOP
    TILELOOP -->|"yes"| LOAD
    LOAD --> SYNC1
    SYNC1 --> COMPUTE
    COMPUTE --> SYNC2
    SYNC2 -->|"more tiles"| TILELOOP
    TILELOOP -->|"done"| WRITE

    style INIT     fill:#2c3e50,color:#ecf0f1,stroke:#1a252f
    style TILELOOP fill:#7d3c98,color:#fff,stroke:#6c3483
    style LOAD     fill:#1f618d,color:#fff,stroke:#154360
    style SYNC1    fill:#c0392b,color:#fff,stroke:#922b21
    style COMPUTE  fill:#1e8449,color:#fff,stroke:#196f3d
    style SYNC2    fill:#c0392b,color:#fff,stroke:#922b21
    style WRITE    fill:#2c3e50,color:#ecf0f1,stroke:#1a252f
```

**Two `__syncthreads()` calls per tile iteration** are required:
1. **After loading** — prevents any thread from reading `A_smem`/`B_smem` before all threads have written their element
2. **After computing** — prevents any thread from overwriting `A_smem`/`B_smem` with the next tile's data before all threads have finished reading this tile

## 4.5 Arithmetic Intensity: Naive vs. Tiled

```mermaid
graph LR
    subgraph NAIVE["🐌 Naive  (no shared memory)"]
        NR["Global reads per output element:\n• K reads from row of A\n• K reads from col of B\nTotal: 2K global reads"]
        NI["Arithmetic Intensity:\n2K FLOPs ÷ (2K × 4 bytes)\n= 0.25 FLOP/byte\n\nGPU roofline: 82 FLOP/byte\n→ 330× below peak ✗"]
    end
    subgraph TILED["🚀 Tiled  (T×T shared memory tile)"]
        TR["Global reads per output element:\n• K/T reads from A (one load per tile)\n• K/T reads from B (one load per tile)\nTotal: 2K/T global reads\nEach element reused T times ✓"]
        TI["Arithmetic Intensity:\n2K FLOPs ÷ (2K/T × 4 bytes)\n= T/8 FLOP/byte\n\nT=16  → 2.0 FLOP/byte\nT=32  → 4.0 FLOP/byte\n→ 8–16× better than naive ✓"]
    end

    style NAIVE fill:#2a0d0d,color:#f1948a,stroke:#e74c3c
    style TILED fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style NR    fill:#c0392b,color:#fff,stroke:#922b21
    style NI    fill:#922b21,color:#f1948a,stroke:#7b241c
    style TR    fill:#1e8449,color:#fff,stroke:#196f3d
    style TI    fill:#145a32,color:#a9dfbf,stroke:#1e8449
```

## 4.6 Benchmarking Matrix Multiply

For a 2048×2048 float32 matrix multiply (~17.2 GFLOP total):

```mermaid
graph LR
    NAIVE2["🐌 Naive\n~5–20 GFLOPS\n\nMemory bandwidth limited\nNo data reuse\nBaseline: 1×"]

    TILED2["⚡ Tiled  (T=16)\n~200–500 GFLOPS\n\nShared memory reuse\nT× bandwidth reduction\n~25–50× over naive"]

    CUBLAS2["🚀 cuBLAS\n~18,000–22,000 GFLOPS\n\nRegister blocking\nVectorized loads (float4)\nWarp specialization\nTensor Cores\n~40–100× over tiled"]

    NAIVE2  -->|"shared memory\nreuse"| TILED2
    TILED2  -->|"register tiling\n+ vectorized loads\n+ Tensor Cores"| CUBLAS2

    style NAIVE2  fill:#c0392b,color:#fff,stroke:#922b21
    style TILED2  fill:#d35400,color:#fff,stroke:#a04000
    style CUBLAS2 fill:#1e8449,color:#fff,stroke:#196f3d
```

The gap between tiled and cuBLAS represents further optimizations we'll see in later chapters: register tiling, vectorized loads, software pipelining, warp specialization.

## 4.7 Beyond Basic Tiling

```mermaid
flowchart TD
    BASIC["Tiled Matmul  (Chapter 04)\nT×T shared memory tiles\nArithmetic intensity: T/8 FLOP/byte"]

    DOUBLE["Double Buffering  (Chapter 06)\nOverlap loading tile N+1\nwhile computing tile N\nHides memory latency"]

    REGBLOCK["Register Blocking\nEach thread computes\na 4×4 or 8×8 sub-tile\nfrom registers (fastest)"]

    VECLOAD["Vectorized Loads  (Chapter 12)\nfloat4 loads 16 bytes/instruction\n4× fewer load instructions"]

    TENSOR["Tensor Cores  (Chapter 12)\nHardware 16×16×16 GEMM unit\n4× throughput over FP32 CUDA cores\ncuBLAS uses these automatically"]

    BASIC --> DOUBLE
    BASIC --> REGBLOCK
    BASIC --> VECLOAD
    REGBLOCK --> TENSOR

    style BASIC    fill:#1f618d,color:#fff,stroke:#154360
    style DOUBLE   fill:#7d3c98,color:#fff,stroke:#6c3483
    style REGBLOCK fill:#d35400,color:#fff,stroke:#a04000
    style VECLOAD  fill:#1e8449,color:#fff,stroke:#196f3d
    style TENSOR   fill:#c0392b,color:#fff,stroke:#922b21
```

See `01_matmul_naive.cu` and `02_matmul_tiled.cu`.

## 4.8 Exercises

1. Run the naive vs tiled benchmarks. Calculate the achieved GFLOPS for each.
2. Experiment with different `TILE` sizes (8, 16, 32). How does performance change? Why might 32 be slower than 16 despite more reuse?
3. What happens if K is not a multiple of TILE? Look at how boundary conditions are handled in the code.
4. The tiled kernel still has non-coalesced access for one of the matrices. Which one? How would you fix it? (Hint: think about how B_smem is loaded.)
5. Implement a version that verifies against a CPU reference for a small matrix (e.g., 64×64).

## 4.9 Key Takeaways

- Matrix multiply is the canonical shared memory optimization problem.
- **Tiling** reduces global memory traffic by factor TILE (dramatically improves arithmetic intensity).
- Threads in a block **cooperate** to load a tile: thread `(ty, tx)` loads element `[ty][tx]` of the tile.
- Both `__syncthreads()` calls (after load and after compute) are **required**.
- A tile size of 16 is common (16×16 = 256 threads, fits within shared memory limits).
- Even optimized tiled code is still far below cuBLAS — many more tricks exist.

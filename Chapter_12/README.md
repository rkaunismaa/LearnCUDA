# Chapter 12: Multi-GPU Programming and Advanced Optimization

## 12.1 Multi-GPU Basics

Modern workstations and servers have multiple GPUs. CUDA supports programming across all of them from a single host process.

```mermaid
graph TD
    subgraph HOST["🖥️ Host Process (single CPU thread)"]
        SET0["cudaSetDevice(0)\\nAll CUDA calls → GPU 0"]
        SET1["cudaSetDevice(1)\\nAll CUDA calls → GPU 1"]
        MERGE["Merge results on CPU\\nor reduce across GPUs"]
    end

    subgraph GPU0["🎮 GPU 0 — RTX 4090"]
        D0["data[0 ... N/2-1]"]
        K0["kernel<<<grid, block>>>()"]
    end

    subgraph GPU1["🎮 GPU 1 — GTX 1050"]
        D1["data[N/2 ... N-1]"]
        K1["kernel<<<grid, block>>>()"]
    end

    SET0 --> D0
    D0   --> K0
    SET1 --> D1
    D1   --> K1
    K0   -->|"D2H result 0"| MERGE
    K1   -->|"D2H result 1"| MERGE

    style HOST fill:#1f618d,color:#fff,stroke:#154360
    style GPU0 fill:#1e8449,color:#fff,stroke:#196f3d
    style GPU1 fill:#7d3c98,color:#fff,stroke:#6c3483
    style SET0 fill:#154360,color:#fff,stroke:#1f618d
    style SET1 fill:#154360,color:#fff,stroke:#1f618d
    style MERGE fill:#d35400,color:#fff,stroke:#a04000
    style D0   fill:#196f3d,color:#fff,stroke:#1e8449
    style K0   fill:#196f3d,color:#fff,stroke:#1e8449
    style D1   fill:#6c3483,color:#fff,stroke:#7d3c98
    style K1   fill:#6c3483,color:#fff,stroke:#7d3c98
```

```c
int deviceCount;
cudaGetDeviceCount(&deviceCount);   // How many GPUs?

// Select GPU for current thread
cudaSetDevice(0);   // All subsequent CUDA calls go to GPU 0
cudaSetDevice(1);   // Now GPU 1

// Check which GPU is current
int current;
cudaGetDevice(&current);
```

Time ≈ single-GPU time / N_gpus (ideal — actual depends on transfer overhead).

## 12.2 Peer-to-Peer (P2P) Access

Without P2P, GPU-to-GPU copies must detour through CPU RAM. With P2P (NVLink or same PCIe switch), GPUs exchange data directly:

```mermaid
graph LR
    subgraph SLOW["Without P2P — through CPU RAM"]
        G0A["GPU 0 VRAM"] -->|"PCIe\\n~16 GB/s"| CRAM["CPU RAM"]
        CRAM -->|"PCIe\\n~16 GB/s"| G1A["GPU 1 VRAM"]
    end

    subgraph FAST["With P2P — NVLink / PCIe direct"]
        G0B["GPU 0 VRAM"] -->|"NVLink\\n~600 GB/s\\n(or PCIe P2P ~32 GB/s)"| G1B["GPU 1 VRAM"]
    end

    style SLOW  fill:#2a0d0d,color:#f1948a,stroke:#e74c3c
    style FAST  fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style G0A   fill:#c0392b,color:#fff,stroke:#922b21
    style CRAM  fill:#c0392b,color:#fff,stroke:#922b21
    style G1A   fill:#c0392b,color:#fff,stroke:#922b21
    style G0B   fill:#1e8449,color:#fff,stroke:#196f3d
    style G1B   fill:#1e8449,color:#fff,stroke:#196f3d
```

```c
// Check if GPU 0 can access GPU 1's memory
int canAccess;
cudaDeviceCanAccessPeer(&canAccess, 0, 1);  // from 0 to 1
if (canAccess) {
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);  // Enable P2P from GPU 0 to GPU 1
    // Now GPU 0 kernels can directly read/write GPU 1 memory
}
```

> Your system has RTX 4090 + GTX 1050 on different PCIe slots without NVLink — P2P via PCIe may or may not be supported. Check with `01_multi_gpu.cu`.

## 12.3 Advanced Optimization Techniques

### Vectorized Memory Loads (float4)

Instead of loading one float per instruction, load 4 floats at once in a single 128-bit transaction:

```diff
  Scalar load — 1 float (32 bits) per instruction:

- float val = data[i];
- // 1 thread × 4 bytes = 4 bytes per load instruction
- // 32 threads × 4 bytes = 128 bytes per warp transaction  (fine, but 4 instructions)

  Vectorized load — 4 floats (128 bits) per instruction:

+ float4 val4 = reinterpret_cast<float4*>(data)[i];
+ float a = val4.x, b = val4.y, c = val4.z, d = val4.w;
+ // 1 thread × 16 bytes = 16 bytes per load instruction
+ // 32 threads × 16 bytes = 512 bytes per warp transaction (4x throughput) ✓
+ // Requirement: array must be 16-byte aligned, size divisible by 4
```

```mermaid
graph LR
    subgraph SCALAR["Scalar Loads — 4 instructions per 4 floats"]
        S0["LD.32  val[0]"]
        S1["LD.32  val[1]"]
        S2["LD.32  val[2]"]
        S3["LD.32  val[3]"]
    end

    subgraph VEC["Vectorized Load — 1 instruction for 4 floats"]
        V0["LD.128  {val[0], val[1], val[2], val[3]}\\n= float4 load (LDG.E.128)"]
    end

    SCALAR -->|"replace with\\n#pragma unroll + float4"| VEC

    style SCALAR fill:#2a0d0d,color:#f1948a,stroke:#e74c3c
    style VEC    fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style S0 fill:#c0392b,color:#fff,stroke:#922b21
    style S1 fill:#c0392b,color:#fff,stroke:#922b21
    style S2 fill:#c0392b,color:#fff,stroke:#922b21
    style S3 fill:#c0392b,color:#fff,stroke:#922b21
    style V0 fill:#1e8449,color:#fff,stroke:#196f3d
```

### `__ldg()` — Read-Only Cache

```diff
  Standard global load — goes through L1/L2 cache:

- float val = data[i];
- // Pollutes L1 cache with data that is never written back
- // Sub-optimal for read-only arrays accessed in irregular patterns

  Read-only cache load — uses the texture cache path:

+ float val = __ldg(&data[i]);
+ // Separate texture/read-only cache — doesn't pollute L1
+ // Better for non-coalesced read-only data (e.g., lookup tables, weights)

+ // Or: compiler auto-uses __ldg when pointer is marked __restrict__:
+ __global__ void kernel(const float* __restrict__ data, ...) { ... }
```

### Instruction-Level Parallelism (ILP)

Each thread computes multiple independent values, letting the GPU pipeline hide instruction latency:

```diff
  ILP=1 — single accumulator, sequential dependency chain:

- for (int i = tid; i < n; i += stride)
-     sum += data[i];           // Each iteration waits for previous add to complete
- // Pipeline stalls every iteration — latency not hidden ✗

  ILP=4 — four independent accumulators, fully pipelined:

+ float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
+ for (int i = tid; i < n; i += stride * 4) {
+     s0 += data[i + 0*stride];   // Independent — can issue all 4 in parallel ✓
+     s1 += data[i + 1*stride];
+     s2 += data[i + 2*stride];
+     s3 += data[i + 3*stride];
+ }
+ float sum = s0 + s1 + s2 + s3;
```

### Loop Unrolling

```c
// Manual unroll — 4 iterations emitted as straight-line code
#pragma unroll 4
for (int k = 0; k < TILE; k++)
    acc += As[ty][k] * Bs[k][tx];

// Full unroll (TILE must be compile-time constant)
#pragma unroll
for (int k = 0; k < TILE; k++)
    acc += As[ty][k] * Bs[k][tx];
```

```mermaid
graph TD
    subgraph OPT["Advanced Optimization Hierarchy"]
        MEM["1. Memory Access\\nfloat4 vectorized loads\\n__ldg read-only cache\\n128-byte coalesced transactions"]
        ILP2["2. Instruction Parallelism\\nILP=4 independent accumulators\\n#pragma unroll\\nhide FMA latency (~4 cycles)"]
        OCC["3. Occupancy\\nenough warps to hide latency\\n(Chapter 7 — register/smem limits)"]
        TC["4. Tensor Cores\\nWMMA API for FP16 MMA\\ncuBLAS for production"]
    end

    MEM --> ILP2 --> OCC --> TC

    style OPT  fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style MEM  fill:#1e8449,color:#fff,stroke:#196f3d
    style ILP2 fill:#d35400,color:#fff,stroke:#a04000
    style OCC  fill:#7d3c98,color:#fff,stroke:#6c3483
    style TC   fill:#c0392b,color:#fff,stroke:#922b21
```

## 12.4 Tensor Cores (WMMA API)

Tensor Cores execute a full 16×16×16 matrix multiply-accumulate (MMA) in a **single warp-level instruction** — the entire warp cooperates:

```mermaid
graph TD
    subgraph TC_ARCH["RTX 4090 Tensor Core Throughput (Ada Lovelace)"]
        INT8["INT8\\n~661 TOPS\\nFastest — inference only"]
        FP16["FP16 / BF16\\n~330 TFLOPS\\nTraining + inference"]
        TF32["TF32\\n~165 TFLOPS\\nFP32 input, FP32-range output"]
        FP64["FP64\\n~1.8 TFLOPS\\nSci compute (weak on Ada)"]
    end

    INT8 --> FP16 --> TF32 --> FP64

    style TC_ARCH fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style INT8    fill:#1e8449,color:#fff,stroke:#196f3d
    style FP16    fill:#1e8449,color:#fff,stroke:#196f3d
    style TF32    fill:#d35400,color:#fff,stroke:#a04000
    style FP64    fill:#c0392b,color:#fff,stroke:#922b21
```

```mermaid
sequenceDiagram
    participant W  as 🔶 Warp (32 threads)
    participant SM as SM — Tensor Core Unit
    participant MEM as Shared / Global Memory

    W  ->> W:   wmma::fill_fragment(frag_c, 0.0f)
    Note over W: All 32 threads declare fragments<br/>frag_a (16×16 FP16), frag_b (16×16 FP16),<br/>frag_c (16×16 FP32 accumulator)

    W  ->> MEM: wmma::load_matrix_sync(frag_a, A_ptr, lda)
    MEM -->> W: 16×16 tile of FP16 values → frag_a

    W  ->> MEM: wmma::load_matrix_sync(frag_b, B_ptr, ldb)
    MEM -->> W: 16×16 tile of FP16 values → frag_b

    W  ->> SM:  wmma::mma_sync(frag_c, frag_a, frag_b, frag_c)
    Note over SM: Single hardware instruction:<br/>frag_c += frag_a × frag_b<br/>16×16×16 = 4096 FMAs in 1 cycle ✓

    W  ->> MEM: wmma::store_matrix_sync(C_ptr, frag_c, ldc, row_major)
```

```c
#include <mma.h>
using namespace nvcuda;

// Fragment declarations (16x16x16 tile)
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>    frag_a;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>    frag_b;
wmma::fragment<wmma::accumulator, 16, 16, 16, float>                  frag_c;

// Initialize accumulator to zero
wmma::fill_fragment(frag_c, 0.0f);

// Load 16x16 tiles from global/shared memory
wmma::load_matrix_sync(frag_a, A_ptr, leading_dim);
wmma::load_matrix_sync(frag_b, B_ptr, leading_dim);

// Execute MMA: frag_c += frag_a * frag_b (one hardware instruction for the whole warp!)
wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

// Store result
wmma::store_matrix_sync(C_ptr, frag_c, leading_dim, wmma::mem_row_major);
```

> **Note**: WMMA is a warp-level API — all 32 threads in the warp must call these functions together. For production: use cuBLAS (already exploits Tensor Cores with auto-tuning).

## 12.5 Checking PTX and Register Usage

```bash
# Compile with register usage report
nvcc -arch=sm_89 -O2 --ptxas-options=-v -o kernel kernel.cu
# Output: ptxas info: Function 'myKernel': 32 registers, 512 bytes smem, ...

# Inspect PTX assembly (virtual ISA)
nvcc -arch=sm_89 -O2 --ptx -o kernel.ptx kernel.cu

# Inspect SASS (actual GPU binary assembly)
cuobjdump --dump-sass kernel
```

```mermaid
flowchart LR
    subgraph COMPILE["Compilation Pipeline"]
        CU2["kernel.cu\\n(CUDA C source)"]
        PTX["kernel.ptx\\n(PTX — virtual ISA)"]
        SASS["kernel binary\\n(SASS — sm_89 native)"]
    end

    subgraph INSPECT["Inspection Tools"]
        PTXAS["--ptxas-options=-v\\nregister count\\nsmem usage\\nspill warnings"]
        PTXDUMP["--ptx flag\\nnvcc output\\nhuman-readable PTX"]
        SASSDUMP["cuobjdump --dump-sass\\nactual executed instructions\\ninstruction scheduling"]
    end

    CU2  -->|"nvcc --ptx"| PTX
    PTX  -->|"ptxas"| SASS
    SASS --> PTXAS
    PTX  --> PTXDUMP
    SASS --> SASSDUMP

    style COMPILE fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style INSPECT fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style CU2     fill:#1f618d,color:#fff,stroke:#154360
    style PTX     fill:#7d3c98,color:#fff,stroke:#6c3483
    style SASS    fill:#1e8449,color:#fff,stroke:#196f3d
    style PTXAS   fill:#d35400,color:#fff,stroke:#a04000
    style PTXDUMP fill:#d35400,color:#fff,stroke:#a04000
    style SASSDUMP fill:#d35400,color:#fff,stroke:#a04000
```

## 12.6 Course Summary — What You've Learned

```mermaid
graph TD
    subgraph FOUNDATIONS["Foundations (Ch 01–04)"]
        C1["Ch 01: GPU Architecture\\nSMs, warps, SIMT, memory hierarchy"]
        C2["Ch 02: Kernels & Thread Indexing\\ngridDim, blockIdx, threadIdx, grid-stride"]
        C3["Ch 03: Memory Optimization\\ncoalescing, shared mem, bank conflicts"]
        C4["Ch 04: Tiled GEMM\\narithmetic intensity, roofline, cooperative load"]
    end

    subgraph INTERMEDIATE["Intermediate (Ch 05–08)"]
        C5["Ch 05: Parallel Reduction\\ntree reduction, warp shuffle, atomics"]
        C6["Ch 06: Streams & Concurrency\\nasync transfers, pinned memory, pipelines"]
        C7["Ch 07: Profiling\\nNsight, CUDA events, occupancy, Little's Law"]
        C8["Ch 08: Unified Memory\\npage migration, prefetch, zero-copy"]
    end

    subgraph ADVANCED["Advanced (Ch 09–12)"]
        C9["Ch 09: Warp Primitives\\nshuffle, ballot, vote, cooperative groups"]
        C10["Ch 10: CUDA Libraries\\ncuBLAS, Thrust, cuRAND, cuFFT"]
        C11["Ch 11: Python CUDA\\nPyTorch, CuPy, RawKernel, AMP"]
        C12["Ch 12: Multi-GPU + Opt\\nP2P, float4, ILP, Tensor Cores (WMMA)"]
    end

    FOUNDATIONS --> INTERMEDIATE --> ADVANCED

    style FOUNDATIONS  fill:#1f618d,color:#fff,stroke:#154360
    style INTERMEDIATE fill:#7d3c98,color:#fff,stroke:#6c3483
    style ADVANCED     fill:#1e8449,color:#fff,stroke:#196f3d
    style C1  fill:#154360,color:#aed6f1,stroke:#1f618d
    style C2  fill:#154360,color:#aed6f1,stroke:#1f618d
    style C3  fill:#154360,color:#aed6f1,stroke:#1f618d
    style C4  fill:#154360,color:#aed6f1,stroke:#1f618d
    style C5  fill:#6c3483,color:#d7bde2,stroke:#7d3c98
    style C6  fill:#6c3483,color:#d7bde2,stroke:#7d3c98
    style C7  fill:#6c3483,color:#d7bde2,stroke:#7d3c98
    style C8  fill:#6c3483,color:#d7bde2,stroke:#7d3c98
    style C9  fill:#196f3d,color:#a9dfbf,stroke:#1e8449
    style C10 fill:#196f3d,color:#a9dfbf,stroke:#1e8449
    style C11 fill:#196f3d,color:#a9dfbf,stroke:#1e8449
    style C12 fill:#196f3d,color:#a9dfbf,stroke:#1e8449
```

## 12.7 Exercises

1. Run `01_multi_gpu.cu` and measure the speedup of dual-GPU vs single-GPU vector add. Is it close to 2x?
2. In `02_vectorized_loads.cu`, verify the float4 and scalar results match. What happens if the array size is not a multiple of 4?
3. Run `03_tensor_cores.cu` and compare FP16 WMMA GFLOPS to FP32 CUDA core GFLOPS. How close to the 4x theoretical speedup do you get?
4. Profile `03_tensor_cores.cu` with `ncu`. Check the "SM Throughput" and "Tensor Active" metrics.
5. Implement a batched WMMA GEMM that processes 16 independent 16×16×16 matrix multiplications in a single kernel.

## 12.8 Key Takeaways

- `cudaSetDevice(n)` selects the active GPU for the current host thread — the simplest multi-GPU model.
- Multi-GPU scaling: split data, launch kernels on each GPU, merge results — near-linear for compute-bound work.
- P2P (NVLink / PCIe direct) eliminates the CPU RAM detour for GPU-to-GPU transfers.
- `float4` vectorized loads reduce instruction count and naturally align to 128-bit memory transactions.
- ILP=4 with independent accumulators lets the GPU pipeline hide instruction latency across iterations.
- WMMA is a **warp-level** API — all 32 threads must call it together; prefer cuBLAS for production.
- For production: use cuBLAS (already exploits Tensor Cores with auto-tuning) rather than raw WMMA.

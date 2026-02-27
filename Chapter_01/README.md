# Chapter 01: GPU Architecture and the CUDA Programming Model

## 1.1 Why GPUs?

Modern CPUs are designed for **latency**: they minimize the time to complete a single task. A CPU has a few powerful cores (typically 8–32), deep out-of-order execution pipelines, large caches, and branch predictors — all aimed at running a single thread as fast as possible.

GPUs are designed for **throughput**: they maximize the total work completed per second. A modern GPU has thousands of smaller, simpler cores. An RTX 4090 has **16,384 CUDA cores**. Each core is weaker than a CPU core, but having thousands of them working in parallel allows the GPU to perform enormous amounts of computation simultaneously.

```mermaid
graph LR
    subgraph CPU["🖥️  CPU — Latency Optimized  (i9-13900K)"]
        C0["Core 0\nALU · FPU · L1$\nBranch Predictor\nOut-of-Order Exec"]
        C1["Core 1\n(same)"]
        Cn["··· 24 cores total\n~5 GHz per core\nFew, but very powerful"]
    end
    subgraph GPU["🎮  GPU — Throughput Optimized  (RTX 4090)"]
        G0["SM 0\n128 CUDA cores"]
        G1["SM 1\n128 CUDA cores"]
        Gn["··· 128 SMs total\n16,384 cores @ ~2.5 GHz\nMany, simpler, massively parallel"]
    end
    style CPU fill:#0d2137,color:#aed6f1,stroke:#2980b9
    style GPU fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style C0 fill:#2471a3,color:#fff,stroke:#1a5276
    style C1 fill:#2471a3,color:#fff,stroke:#1a5276
    style Cn fill:#154360,color:#85c1e9,stroke:#1a6091
    style G0 fill:#1e8449,color:#fff,stroke:#196f3d
    style G1 fill:#1e8449,color:#fff,stroke:#196f3d
    style Gn fill:#145a32,color:#82e882,stroke:#1e8449
```

```
Die area breakdown (approximate):
  CPU: ~50% cache  |  ~30% control logic  |  ~20% compute
  GPU: ~80% compute (ALUs)  |  ~15% memory ctrl  |  ~5% control
```

The key insight: many computational problems — especially in graphics, machine learning, and scientific computing — involve performing the **same operation on large arrays of data**. This is called **data parallelism**, and GPUs exploit it perfectly.

## 1.2 GPU Hardware Architecture

Understanding the hardware hierarchy helps you write efficient CUDA code.

### Streaming Multiprocessors (SMs)

A GPU die is organized into **Streaming Multiprocessors (SMs)**. The RTX 4090 has **128 SMs**.

```mermaid
graph TB
    subgraph SM["⚙️  Streaming Multiprocessor (SM) — 1 of 128 on RTX 4090 Ada Lovelace"]
        subgraph SCHED["🟢  Warp Schedulers (4×)"]
            WS["Issue 1 warp instruction per clock\nManage up to 48 resident warps (1,536 threads)"]
        end
        subgraph EXEC["🔴  Execution Units"]
            CC["🔵  128 CUDA Cores — FP32 / INT32"]
            TC["🟣  4× Tensor Cores — FP16 matrix multiply (AI/ML)"]
            RT["⚪  1× RT Core — Ray tracing BVH traversal"]
            SF["🟠  4× SFUs — sin / cos / sqrt / rcp"]
        end
        subgraph ONCHIP["🟣  On-Chip Memory"]
            REG["📦  Register File: 65,536 × 32-bit = 256 KB\n(fastest storage — private per thread)"]
            SHM["⚡  Shared Memory / L1 Cache: 128 KB\n(user-controlled scratchpad — shared per block)"]
        end
    end
    SCHED --> EXEC
    style SM fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style SCHED fill:#0d2a0d,color:#a9dfbf,stroke:#27ae60
    style EXEC fill:#2a0d0d,color:#f1948a,stroke:#e74c3c
    style ONCHIP fill:#1a0d2a,color:#d2b4de,stroke:#8e44ad
    style WS fill:#1e8449,color:#fff,stroke:#196f3d
    style CC fill:#2471a3,color:#fff,stroke:#1a5276
    style TC fill:#7d3c98,color:#fff,stroke:#6c3483
    style RT fill:#566573,color:#fff,stroke:#4d5656
    style SF fill:#ca6f1e,color:#fff,stroke:#b9770e
    style REG fill:#1a5276,color:#fff,stroke:#154360
    style SHM fill:#6c3483,color:#fff,stroke:#5b2c6f
```

> 128 SMs × 128 CUDA cores = **16,384 CUDA cores** on the RTX 4090

### The Warp: The Fundamental Execution Unit

The GPU does **not** execute one thread at a time. Threads are grouped into **warps** of 32 threads. All 32 threads in a warp execute the **same instruction simultaneously** — this is called **SIMT** (Single Instruction, Multiple Threads).

When threads within a warp take different paths (branch divergence), the GPU must serialize them, reducing efficiency:

```diff
  ── No Branch Divergence — efficient (all 32 threads take the same path) ──

+ Thread  0:  [instr0][instr1][instr2][instr3][instr4]  ACTIVE  5/5 cycles ✓
+ Thread  1:  [instr0][instr1][instr2][instr3][instr4]  ACTIVE  5/5 cycles ✓
+ Thread  2:  [instr0][instr1][instr2][instr3][instr4]  ACTIVE  5/5 cycles ✓
  ...
+ Thread 31:  [instr0][instr1][instr2][instr3][instr4]  ACTIVE  5/5 cycles ✓

  → 5 cycles total | 100% warp efficiency ✓


  ── Branch Divergence: if (threadIdx.x < 16) { doA(); } else { doB(); } ──

  Pass 1: Branch A executes — threads 16–31 are MASKED (idle)
+ Thread  0:  [A0][A1][A2][A3][──][──][──]  active (branch A)
+ Thread 15:  [A0][A1][A2][A3][──][──][──]  active (branch A)
- Thread 16:  [──][──][──][──][──][──][──]  MASKED — waiting
- Thread 31:  [──][──][──][──][──][──][──]  MASKED — waiting

  Pass 2: Branch B executes — threads 0–15 are MASKED (idle)
- Thread  0:  [──][──][──][──][──][──][──]  MASKED — waiting
- Thread 15:  [──][──][──][──][──][──][──]  MASKED — waiting
+ Thread 16:  [──][──][──][──][B0][B1][B2]  active (branch B)
+ Thread 31:  [──][──][──][──][B0][B1][B2]  active (branch B)

  → 4 + 3 = 7 cycles total | ~57% warp efficiency ✗
  Rule: keep all 32 threads in a warp on the same code path
```

### Memory Hierarchy

The memory system has multiple levels. Moving data to faster memory closer to the compute cores is the primary CUDA optimization strategy.

```mermaid
graph TB
    R["⚡ Registers\n256 KB per SM  |  ~1 cycle latency\nBandwidth ≈ 17,000 GB/s\nPer-thread — compiler-managed"]
    S["🔥 Shared Memory / L1 Cache\n128 KB per SM  |  ~5–30 cycles\nBandwidth ≈ 19,000 GB/s\nPer-block — programmer-managed scratchpad"]
    L["💡 L2 Cache\n72 MB on-chip  |  ~100–200 cycles\nBandwidth ≈ 7,000 GB/s\nShared across all 128 SMs"]
    G["💾 Global Memory (VRAM)\n24 GB GDDR6X  |  ~200–600 cycles\nBandwidth ≈ 1,008 GB/s\nAll threads can read/write"]
    P["🐌 System RAM (CPU)\n~64 GB+ DDR5  |  ~1,000+ cycles\nBandwidth ≈ 64 GB/s via PCIe\nAccessed via cudaMemcpy — very slow"]

    R --> S --> L --> G --> P

    style R fill:#c0392b,color:#fff,stroke:#922b21
    style S fill:#d35400,color:#fff,stroke:#a04000
    style L fill:#c8a000,color:#1a1a1a,stroke:#b7950b
    style G fill:#1f618d,color:#fff,stroke:#154360
    style P fill:#6c3483,color:#fff,stroke:#5b2c6f
```

| Level | Location | Latency | Size | Shared? |
|-------|----------|---------|------|---------|
| Registers | Inside SM | ~1 cycle | 64K per SM | Per-thread |
| Shared Memory | Inside SM | ~5-30 cycles | Up to 100 KB per SM | Per-block |
| L1 Cache | Inside SM | ~20-50 cycles | 128 KB per SM | Per-SM |
| L2 Cache | On-chip | ~100-200 cycles | 72 MB (4090) | All SMs |
| Global Memory | GDDR6X VRAM | ~200-600 cycles | 24 GB (4090) | All threads |
| System RAM | CPU RAM | ~1000+ cycles | As configured | Via PCIe |

## 1.3 CUDA: The Programming Model

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform. It lets you write C/C++ code that runs on the GPU.

### Key Terminology

| CUDA Term | Hardware Equivalent |
|-----------|-------------------|
| Thread | One CUDA core running one instance of your function |
| Warp | 32 threads executing together (hardware level) |
| Block | User-defined group of threads (share shared memory) |
| Grid | All blocks launched for one kernel call |
| Kernel | A function that runs on the GPU |
| Host | The CPU and its memory |
| Device | The GPU and its memory |

### The Thread Hierarchy

```mermaid
graph TD
    K["kernel&lt;&lt;&lt; gridDim, blockDim &gt;&gt;&gt;()"]

    subgraph GRID["GRID — all blocks for one kernel call\nBlocks can run on any SM in any order"]
        B00["Block 0,0\nblockIdx={0,0}"]
        B10["Block 1,0\nblockIdx={1,0}"]
        B20["Block 2,0\nblockIdx={2,0}"]
        Bdot["Block N,M\n..."]
    end

    subgraph BLK["BLOCK — e.g., Block 0,0\nThreads share Shared Memory\nAll run on the same SM"]
        T00["Thread 0,0\nthreadIdx={0,0}"]
        T10["Thread 1,0\nthreadIdx={1,0}"]
        T01["Thread 0,1\nthreadIdx={0,1}"]
        Tdot["Thread ...\n..."]
    end

    K --> GRID
    B00 -->|"zoom in"| BLK

    style K fill:#2c3e50,color:#ecf0f1,stroke:#1a252f
    style GRID fill:#0d1b2a,color:#aed6f1,stroke:#2980b9
    style BLK fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style B00 fill:#e74c3c,color:#fff,stroke:#c0392b
    style B10 fill:#2471a3,color:#fff,stroke:#1a5276
    style B20 fill:#2471a3,color:#fff,stroke:#1a5276
    style Bdot fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style T00 fill:#1e8449,color:#fff,stroke:#196f3d
    style T10 fill:#1e8449,color:#fff,stroke:#196f3d
    style T01 fill:#1e8449,color:#fff,stroke:#196f3d
    style Tdot fill:#145a32,color:#82e882,stroke:#1e8449
```

Every thread has a unique identity via:
- `threadIdx.x/y/z` — position within its block
- `blockIdx.x/y/z` — position of the block in the grid
- `blockDim.x/y/z` — size of each block
- `gridDim.x/y/z` — size of the grid

### Concrete Thread Index Calculation

```
Example: 1D grid of 3 blocks, each with 4 threads

Launch: kernel<<<3, 4>>>()        gridDim.x = 3
                                  blockDim.x = 4

Block 0          Block 1          Block 2
┌──┬──┬──┬──┐   ┌──┬──┬──┬──┐   ┌──┬──┬──┬──┐
│T0│T1│T2│T3│   │T0│T1│T2│T3│   │T0│T1│T2│T3│
└──┴──┴──┴──┘   └──┴──┴──┴──┘   └──┴──┴──┴──┘
threadIdx.x:
 0  1  2  3      0  1  2  3      0  1  2  3
blockIdx.x:
 0  0  0  0      1  1  1  1      2  2  2  2

Global index = blockIdx.x * blockDim.x + threadIdx.x
               0*4+0=0             1*4+0=4             2*4+0=8
               0*4+1=1             1*4+1=5             2*4+1=9
               0*4+2=2             1*4+2=6             2*4+2=10
               0*4+3=3             1*4+3=7             2*4+3=11

Array: [A0][A1][A2][A3][A4][A5][A6][A7][A8][A9][A10][A11]
         ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲    ▲    ▲
       T0  T1  T2  T3  T4  T5  T6  T7  T8  T9  T10  T11
```

### A CUDA Program's Execution Flow

```mermaid
sequenceDiagram
    participant CPU as 🖥️ HOST (CPU)
    participant BUS as ⚡ PCIe Bus
    participant GPU as 🎮 DEVICE (GPU)

    CPU->>GPU: ① cudaMalloc() — allocate device memory (VRAM)
    activate GPU

    CPU->>BUS: ② cudaMemcpy(d_A, h_A, H→D)
    BUS->>GPU: data transferred (~10 ms for 256 MB)
    deactivate GPU

    CPU-)GPU: ③ kernel<<<grid,block>>>() — ASYNC launch
    Note over CPU: CPU returns immediately and<br/>continues running (non-blocking!)
    activate GPU
    Note over GPU: Thousands of threads run<br/>in parallel across all SMs

    CPU->>GPU: ④ cudaDeviceSynchronize() — barrier
    GPU-->>CPU: kernel complete ✓
    deactivate GPU

    CPU->>BUS: ⑤ cudaMemcpy(h_C, d_C, D→H)
    BUS->>CPU: results transferred (~10 ms)

    CPU->>GPU: ⑥ cudaFree() — release VRAM
```

## 1.4 Your First CUDA Program

See `01_hello_cuda.cu` — prints from both CPU and GPU threads.

See `02_device_info.cu` — queries and prints detailed GPU hardware information.

## 1.5 Compiling CUDA Code

CUDA source files use the `.cu` extension and are compiled with `nvcc`:

```bash
nvcc -o hello 01_hello_cuda.cu
./hello
```

```mermaid
flowchart LR
    SRC["📄 mykernel.cu\nCUDA C/C++ source"]
    NVCC["⚙️ nvcc\nNVIDIA CUDA Compiler"]
    PTX["📋 PTX\nVirtual ISA\n(portable bytecode)"]
    SASS["💻 SASS\nGPU machine code\n(sm_89 binary)"]
    HOST["🖥️ Host C++ code\n(CPU portion)"]
    GPP["⚙️ g++\nHost C++ Compiler"]
    BIN["🚀 ./binary\nFinal executable"]

    SRC --> NVCC
    NVCC --> PTX
    NVCC --> HOST
    PTX --> SASS
    HOST --> GPP
    SASS --> BIN
    GPP --> BIN

    style SRC fill:#2c3e50,color:#ecf0f1,stroke:#1a252f
    style NVCC fill:#7d3c98,color:#fff,stroke:#6c3483
    style PTX fill:#1f618d,color:#fff,stroke:#154360
    style SASS fill:#1e8449,color:#fff,stroke:#196f3d
    style HOST fill:#2c3e50,color:#ecf0f1,stroke:#1a252f
    style GPP fill:#7d3c98,color:#fff,stroke:#6c3483
    style BIN fill:#c0392b,color:#fff,stroke:#922b21
```

Common `nvcc` flags:

| Flag | Purpose |
|------|---------|
| `-o <name>` | Output binary name |
| `-arch=sm_89` | Target compute capability (89 = RTX 4090) |
| `-arch=sm_61` | Target CC 6.1 (GTX 1050) |
| `-G` | Enable device-side debugging |
| `-lineinfo` | Embed source line info (for profilers) |
| `-O2` | Optimization level |
| `--use_fast_math` | Use faster (slightly less precise) math ops |

## 1.6 CUDA Error Checking

CUDA API functions return `cudaError_t`. **Always check for errors** — silent failures are the #1 debugging headache in CUDA.

```c
cudaError_t err = cudaMalloc(&d_ptr, size);
if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
    exit(1);
}
```

We define a convenient macro `CUDA_CHECK` in our examples:

```c
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = (call);                                      \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)
```

## 1.7 GPU vs CPU: When to Use Each

```mermaid
flowchart TD
    Q{"🤔 Is my problem\na good GPU fit?"}

    Q -->|"Same op repeated\nover large dataset\nN &gt; ~10,000 items"| PARA
    Q -->|"Sequential logic,\ncomplex branches,\nor small dataset"| CPU

    PARA -->|"Few data\ndependencies\nbetween items"| GPU
    PARA -->|"Heavy inter-item\ndependencies or\nrecursion"| CPU

    GPU["🚀 USE THE GPU\n───────────────────\n• Matrix / tensor ops\n• Neural network layers\n• Image / video processing\n• FFT and signal processing\n• Monte Carlo simulation\n• Physics / fluid dynamics\n• Sorting large arrays"]

    CPU["🖥️ USE THE CPU\n───────────────────\n• OS calls / file I/O\n• Parsing text / JSON\n• Complex decision trees\n• Pointer-chasing structs\n• Recursive algorithms\n• Small datasets\n• Orchestrating GPU work"]

    style Q fill:#7d3c98,color:#fff,stroke:#6c3483
    style PARA fill:#1e8449,color:#fff,stroke:#196f3d
    style GPU fill:#145a32,color:#a9dfbf,stroke:#1e8449
    style CPU fill:#1a5276,color:#aed6f1,stroke:#154360
```

## 1.8 Exercises

1. Compile and run `01_hello_cuda.cu`. Notice that GPU output order is non-deterministic — why?
2. Compile and run `02_device_info.cu`. Note the warp size, max threads per block, and SM count for your GPU.
3. Modify `01_hello_cuda.cu` to launch 4 blocks of 8 threads each. How many lines of GPU output do you see?
4. Look up the compute capability of your GPU on the [CUDA GPU list](https://developer.nvidia.com/cuda-gpus). What new features does your CC enable?

## 1.9 Key Takeaways

- GPUs have thousands of simple cores optimized for throughput over latency.
- The fundamental hardware unit is the **warp** (32 threads executing in lockstep).
- **Branch divergence** within a warp serializes execution — keep threads on the same code path.
- CUDA organizes threads into a hierarchy: **thread → block → grid**.
- Global thread index: `blockIdx.x * blockDim.x + threadIdx.x`
- Memory is hierarchical: registers → shared memory → L2 → global DRAM.
- Every CUDA program follows: allocate → copy to GPU → kernel launch → sync → copy back → free.
- **Always check CUDA error codes.**

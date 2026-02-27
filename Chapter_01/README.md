# Chapter 01: GPU Architecture and the CUDA Programming Model

## 1.1 Why GPUs?

Modern CPUs are designed for **latency**: they minimize the time to complete a single task. A CPU has a few powerful cores (typically 8–32), deep out-of-order execution pipelines, large caches, and branch predictors — all aimed at running a single thread as fast as possible.

GPUs are designed for **throughput**: they maximize the total work completed per second. A modern GPU has thousands of smaller, simpler cores. An RTX 4090 has **16,384 CUDA cores**. Each core is weaker than a CPU core, but having thousands of them working in parallel allows the GPU to perform enormous amounts of computation simultaneously.

```
┌─────────────────────────────────┐   ┌──────────────────────────────────────┐
│           CPU (i9-13900K)       │   │         GPU (RTX 4090)               │
│                                 │   │                                      │
│  ┌──────┐ ┌──────┐ ┌──────┐    │   │  ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐  │
│  │Core 0│ │Core 1│ │Core 2│    │   │  └──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘  │
│  │ ALU  │ │ ALU  │ │ ALU  │    │   │  ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐  │
│  │ FPU  │ │ FPU  │ │ FPU  │    │   │  └──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘  │
│  │ L1$  │ │ L1$  │ │ L1$  │    │   │  ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐  │
│  └──────┘ └──────┘ └──────┘    │   │  └──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘  │
│  ...24 total cores...           │   │  ... 16,384 CUDA cores (128 SMs) ... │
│                                 │   │                                      │
│  Design goal: LOW LATENCY       │   │  Design goal: HIGH THROUGHPUT        │
│  Fast single-threaded tasks     │   │  Massive data-parallel workloads     │
│  ~24 cores × ~5 GHz             │   │  16,384 cores × ~2.5 GHz            │
└─────────────────────────────────┘   └──────────────────────────────────────┘

 Die area breakdown (approximate):
 CPU: ~50% cache, ~30% control logic, ~20% compute
 GPU: ~80% compute (ALUs), ~15% memory, ~5% control
```

The key insight: many computational problems — especially in graphics, machine learning, and scientific computing — involve performing the **same operation on large arrays of data**. This is called **data parallelism**, and GPUs exploit it perfectly.

## 1.2 GPU Hardware Architecture

Understanding the hardware hierarchy helps you write efficient CUDA code.

### Streaming Multiprocessors (SMs)

A GPU die is organized into **Streaming Multiprocessors (SMs)**, sometimes called "Compute Units" on AMD hardware. The RTX 4090 has **128 SMs**.

```
┌─────────────────────────────────────────────────────────────────┐
│                   Streaming Multiprocessor (SM)                  │
│                        RTX 4090 Ada Lovelace                    │
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │  Warp          │  │  Warp          │  │  Warp          │    │
│  │  Scheduler 0   │  │  Scheduler 1   │  │  Scheduler 2   │    │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘    │
│          │                   │                   │              │
│  ┌───────▼───────────────────▼───────────────────▼──────────┐  │
│  │              Dispatch Units                               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 CUDA Cores (FP32/INT)                    │   │
│  │  [ALU][ALU][ALU][ALU][ALU][ALU][ALU][ALU]  × 16 rows   │   │
│  │                 128 CUDA cores total                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────┐  ┌───────────────┐  ┌─────────────────┐  │
│  │  Tensor Cores    │  │  RT Cores     │  │  SFUs (sin/cos) │  │
│  │  (FP16 matmul)   │  │  (ray trace)  │  │  4 units        │  │
│  └──────────────────┘  └───────────────┘  └─────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │    Register File: 65,536 × 32-bit registers (256 KB)     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │    Shared Memory / L1 Cache  (128 KB, configurable)      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

   128 SMs on RTX 4090  →  128 × 128 = 16,384 CUDA cores total
```

### The Warp: The Fundamental Execution Unit

The GPU does **not** execute one thread at a time. Instead, threads are grouped into **warps** of 32 threads. All 32 threads in a warp execute the **same instruction simultaneously** — this is called **SIMT** (Single Instruction, Multiple Threads).

```
SIMT Execution — No Branch Divergence (efficient):
                    Cycle  1  2  3  4  5
                           │  │  │  │  │
Thread  0: instr A ────────▼──▼──▼──▼──▼──▶
Thread  1: instr A ────────▼──▼──▼──▼──▼──▶
Thread  2: instr A ────────▼──▼──▼──▼──▼──▶
  ...                    (all 32 threads in lockstep)
Thread 31: instr A ────────▼──▼──▼──▼──▼──▶

Cost: 5 cycles for the entire warp ✓


SIMT Execution — Branch Divergence (inefficient):
if (threadIdx.x < 16) { branch A } else { branch B }

Cycle:    1   2   3   4   5   6   7   8
          │   │   │   │   │   │   │   │
Thread  0─▼───▼───▼─(branch A active)──▶  ✓ active
Thread  1─▼───▼───▼─(branch A active)──▶  ✓ active
  ...
Thread 15─▼───▼───▼─(branch A active)──▶  ✓ active
Thread 16─▼───▼───▼─ IDLE ─── IDLE ───▶  ✗ masked
  ...
Thread 31─▼───▼───▼─ IDLE ─── IDLE ───▶  ✗ masked
                     then branch B runs:
Thread  0─(IDLE)─(IDLE)─▼───▼───▼─────▶  ✗ masked
  ...
Thread 15─(IDLE)─(IDLE)─▼───▼───▼─────▶  ✗ masked
Thread 16─(IDLE)─(IDLE)─▼───▼───▼─────▶  ✓ active
  ...
Thread 31─(IDLE)─(IDLE)─▼───▼───▼─────▶  ✓ active

Cost: 5 + 3 = 8 cycles (serialized) — 60% efficiency ✗
```

### Memory Hierarchy

From fastest/smallest to slowest/largest:

```
Closer to cores ──────────────────────────────────── Farther from cores
Faster/smaller                                         Slower/larger

  Registers       Shared Mem        L2 Cache         Global VRAM
  ┌────────┐      ┌────────┐       ┌────────┐        ┌──────────┐
  │256 KB  │      │128 KB  │       │ 72 MB  │        │  24 GB   │
  │per SM  │      │per SM  │       │on-chip │        │ GDDR6X   │
  └────────┘      └────────┘       └────────┘        └──────────┘
  ~1 cycle        ~5-30 cycles     ~100-200 cycles   ~200-600 cycles
  ~17 TB/s        ~19 TB/s         ~7 TB/s           ~1,008 GB/s

  ▲ Use as much as possible!                              ▲ Unavoidable
                                                            bottleneck

Relative bandwidth (higher = more data per second):
Registers  ████████████████████████████████████████  ~17,000 GB/s
Shared Mem ████████████████████████████████████████  ~19,000 GB/s
L2 Cache   ████████████████████                       ~7,000 GB/s
Global Mem ████                                        ~1,008 GB/s
PCIe (RAM) ▌                                              ~64 GB/s
```

| Level | Location | Latency | Size | Shared? |
|-------|----------|---------|------|---------|
| Registers | Inside SM | ~1 cycle | 64K per SM | Per-thread |
| Shared Memory | Inside SM | ~5-30 cycles | Up to 100 KB per SM | Per-block |
| L1 Cache | Inside SM | ~20-50 cycles | 128 KB per SM | Per-SM |
| L2 Cache | On-chip | ~100-200 cycles | 72 MB (4090) | All SMs |
| Global Memory | GDDR6X VRAM | ~200-600 cycles | 24 GB (4090) | All threads |
| System RAM | CPU RAM | ~1000+ cycles | As configured | Via PCIe |

This hierarchy is central to CUDA optimization — you'll spend much of your time moving data closer to the compute cores.

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

```
Kernel Launch: myKernel<<<grid, block>>>()
                                │
┌───────────────────────────────▼──────────────────────────────────┐
│                           GRID                                    │
│          (example: gridDim = {4 blocks wide, 3 blocks tall})     │
│                                                                   │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────┐  │
│   │  Block(0,0) │  │  Block(1,0) │  │  Block(2,0) │  │B(3,0) │  │
│   │  ┌─┬─┬─┬─┐ │  │  ┌─┬─┬─┬─┐ │  │  ┌─┬─┬─┬─┐ │  │  ...  │  │
│   │  │0│1│2│3│ │  │  │0│1│2│3│ │  │  │0│1│2│3│ │  │       │  │
│   │  ├─┼─┼─┼─┤ │  │  ├─┼─┼─┼─┤ │  │  ├─┼─┼─┼─┤ │  │       │  │
│   │  │4│5│6│7│ │  │  │4│5│6│7│ │  │  │4│5│6│7│ │  │       │  │
│   │  └─┴─┴─┴─┘ │  │  └─┴─┴─┴─┘ │  │  └─┴─┴─┴─┘ │  └───────┘  │
│   │blockIdx=(0,0)│  │blockIdx=(1,0)│  │blockIdx=(2,0)│            │
│   └─────────────┘  └─────────────┘  └─────────────┘            │
│   ┌─────────────┐  ┌─────────────┐  ...                         │
│   │  Block(0,1) │  │  Block(1,1) │                              │
│   │  ...        │  │  ...        │                              │
│   └─────────────┘  └─────────────┘                              │
└──────────────────────────────────────────────────────────────────┘

Each block has its own shared memory.
Blocks can run on any SM in any order — the GPU scheduler decides.
```

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

This maps perfectly to array indices:
Array: [A0][A1][A2][A3][A4][A5][A6][A7][A8][A9][A10][A11]
         ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲    ▲    ▲
       T0  T1  T2  T3  T4  T5  T6  T7  T8  T9  T10  T11
      (B0) (B0)(B0)(B0)(B1)(B1)(B1)(B1)(B2)(B2) (B2) (B2)
```

Every thread has a unique identity via:
- `threadIdx.x`, `threadIdx.y`, `threadIdx.z` — position within its block
- `blockIdx.x`, `blockIdx.y`, `blockIdx.z` — position of its block in the grid
- `blockDim.x/y/z` — dimensions of each block
- `gridDim.x/y/z` — dimensions of the grid

### A CUDA Program's Execution Flow

```
HOST (CPU)                                    DEVICE (GPU)
──────────────────────────────────────────────────────────────
                          PCIe Bus (up to 64 GB/s)
Time │
     │
  ①  ▼  cudaMalloc(d_A, size) ──────────────▶ Allocate VRAM
     │
  ②  ▼  cudaMemcpy(d_A, h_A, ─────────────── Copy H→D ──────▶
     │             H2D)                       ~1-10 ms
     │
  ③  ▼  kernel<<<grid,block>>>() ──────────── Launch kernel
     │  (returns immediately!)                │
     │  ...host can do other work here...     │  Thousands of
     │                                        │  threads run
     │                                        │  in parallel
     │                                        │
  ④  ▼  cudaDeviceSynchronize() ─ wait ──────◀ (kernel done)
     │
  ⑤  ▼  cudaMemcpy(h_C, d_C, ◀──────────────── Copy D→H ◀──
     │             D2H)                        ~1-10 ms
     │
  ⑥  ▼  cudaFree(d_A) ────────────────────── Free VRAM
     │
     ▼  Process results on CPU

Key insight: The kernel launch (③) is ASYNCHRONOUS — the CPU
continues running while the GPU works. cudaDeviceSynchronize()
is the barrier that waits for GPU completion.
```

## 1.4 Your First CUDA Program

See `01_hello_cuda.cu` — this prints from both CPU and GPU threads.

See `02_device_info.cu` — this queries and prints detailed GPU hardware information.

## 1.5 Compiling CUDA Code

CUDA source files use the `.cu` extension and are compiled with `nvcc` (NVIDIA's CUDA compiler):

```bash
nvcc -o hello 01_hello_cuda.cu
./hello
```

```
nvcc Compilation Pipeline:

  mykernel.cu  ──▶  nvcc  ──┬──▶ PTX (virtual ISA)  ──▶ SASS (machine code)
                             │                              │
                             └──▶ host C++ code  ──▶ g++   │
                                                     │      │
                                                     └──────┴──▶  ./binary
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

For our RTX 4090, always compile with at least `-arch=sm_89` to get full performance. For code that must also run on the GTX 1050, use `-arch=sm_61`.

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

```
                        ┌─────────────────────────────────────┐
                        │        Problem characteristics       │
                        └──────────────┬──────────────────────┘
                                       │
                   ┌───────────────────┴─────────────────────┐
                   │                                         │
         Highly parallel?                            Sequential or
         Same op on many data?                       branchy logic?
                   │                                         │
                   ▼                                         ▼
         ┌─────────────────┐                       ┌─────────────────┐
         │   Use the GPU   │                       │   Use the CPU   │
         │  Matrix ops     │                       │  OS / I/O       │
         │  Image filters  │                       │  File parsing   │
         │  Neural nets    │                       │  Complex trees  │
         │  FFT / signals  │                       │  Recursion      │
         │  Monte Carlo    │                       │  Small datasets │
         └─────────────────┘                       └─────────────────┘

Rule of thumb: GPU wins when N > ~10,000 independent operations
               GPU loses when dependencies prevent parallelism
```

## 1.8 Exercises

1. Compile and run `01_hello_cuda.cu`. Notice that GPU output order is non-deterministic — why?
2. Compile and run `02_device_info.cu`. Note the warp size, max threads per block, and SM count for your GPU.
3. Modify `01_hello_cuda.cu` to launch 4 blocks of 8 threads each. How many lines of GPU output do you see?
4. Look up the compute capability of your GPU on the [CUDA GPU list](https://developer.nvidia.com/cuda-gpus). What new features does your CC enable?

## 1.9 Key Takeaways

- GPUs have thousands of simple cores optimized for throughput over latency.
- The fundamental hardware unit is the **warp** (32 threads executing in lockstep).
- **Branch divergence** within a warp serializes execution — keep threads uniform.
- CUDA organizes threads into a hierarchy: **thread → block → grid**.
- Global thread index: `blockIdx.x * blockDim.x + threadIdx.x`
- Memory is hierarchical: registers → shared memory → L2 → global DRAM.
- Every CUDA program follows: allocate → copy to GPU → kernel launch → sync → copy back → free.
- **Always check CUDA error codes.**

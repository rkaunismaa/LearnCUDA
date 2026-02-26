# Chapter 01: GPU Architecture and the CUDA Programming Model

## 1.1 Why GPUs?

Modern CPUs are designed for **latency**: they minimize the time to complete a single task. A CPU has a few powerful cores (typically 8–32), deep out-of-order execution pipelines, large caches, and branch predictors — all aimed at running a single thread as fast as possible.

GPUs are designed for **throughput**: they maximize the total work completed per second. A modern GPU has thousands of smaller, simpler cores. An RTX 4090 has **16,384 CUDA cores**. Each core is weaker than a CPU core, but having thousands of them working in parallel allows the GPU to perform enormous amounts of computation simultaneously.

```
CPU: Few powerful cores — minimize latency per task
    [Core][Core][Core][Core]  <- 4-32 cores, complex

GPU: Many simple cores — maximize throughput
    [c][c][c][c][c][c][c][c]
    [c][c][c][c][c][c][c][c]  <- thousands of cores, simpler
    [c][c][c][c][c][c][c][c]
    [c][c][c][c][c][c][c][c]
```

The key insight: many computational problems — especially in graphics, machine learning, and scientific computing — involve performing the **same operation on large arrays of data**. This is called **data parallelism**, and GPUs exploit it perfectly.

## 1.2 GPU Hardware Architecture

Understanding the hardware hierarchy helps you write efficient CUDA code.

### Streaming Multiprocessors (SMs)

A GPU die is organized into **Streaming Multiprocessors (SMs)**, sometimes called "Compute Units" on AMD hardware. The RTX 4090 has **128 SMs**.

Each SM contains:
- A set of CUDA cores (also called "shader processors" or "SPs")
- Warp schedulers
- Register file (a large pool of fast registers)
- Shared memory / L1 cache (on-chip, very fast)
- Special function units (SFUs) for sin/cos/etc.

```
GPU
└── SM 0
│   ├── 128 CUDA Cores (RTX 4090: 128 per SM)
│   ├── 4 Warp Schedulers
│   ├── Register File (65536 x 32-bit registers)
│   ├── Shared Memory / L1 Cache (128 KB)
│   └── Texture Units, SFUs, ...
└── SM 1
│   └── (same structure)
└── ...
└── SM 127
    └── (same structure)
```

### The Warp: The Fundamental Execution Unit

The GPU does **not** execute one thread at a time. Instead, threads are grouped into **warps** of 32 threads. All 32 threads in a warp execute the **same instruction simultaneously** — this is called **SIMT** (Single Instruction, Multiple Threads).

This is like having a squad of 32 soldiers who all perform the same command together. If they need to do different things (branch divergence), some must wait while others execute — efficiency drops.

### Memory Hierarchy

From fastest/smallest to slowest/largest:

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
Grid (one kernel launch)
└── Block (0,0)  Block (1,0)  Block (2,0)
│   └── Thread(0,0) Thread(1,0) ... Thread(31,0)
│       Thread(0,1) Thread(1,1) ... Thread(31,1)
│       ...
└── Block (0,1)  Block (1,1)  ...
    └── ...
```

Every thread has a unique identity via:
- `threadIdx.x`, `threadIdx.y`, `threadIdx.z` — position within its block
- `blockIdx.x`, `blockIdx.y`, `blockIdx.z` — position of its block in the grid
- `blockDim.x/y/z` — dimensions of each block
- `gridDim.x/y/z` — dimensions of the grid

### A CUDA Program's Structure

```
┌─────────────────────────────────────────────────┐
│                  HOST (CPU)                     │
│                                                 │
│  1. Allocate memory on GPU (cudaMalloc)         │
│  2. Copy data CPU → GPU (cudaMemcpy)            │
│  3. Launch kernel (<<<grid, block>>>)           │
│  4. Wait for GPU to finish (cudaDeviceSynchronize) │
│  5. Copy results GPU → CPU (cudaMemcpy)         │
│  6. Free GPU memory (cudaFree)                  │
└───────────────────────┬─────────────────────────┘
                        │ PCIe / NVLink
┌───────────────────────▼─────────────────────────┐
│                  DEVICE (GPU)                   │
│                                                 │
│  Kernel runs with thousands of threads          │
│  Each thread executes the kernel function       │
│  with unique threadIdx / blockIdx               │
└─────────────────────────────────────────────────┘
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

## 1.7 Exercises

1. Compile and run `01_hello_cuda.cu`. Notice that GPU output order is non-deterministic — why?
2. Compile and run `02_device_info.cu`. Note the warp size, max threads per block, and SM count for your GPU.
3. Modify `01_hello_cuda.cu` to launch 4 blocks of 8 threads each. How many lines of GPU output do you see?
4. Look up the compute capability of your GPU on the [CUDA GPU list](https://developer.nvidia.com/cuda-gpus). What new features does your CC enable?

## 1.8 Key Takeaways

- GPUs have thousands of simple cores optimized for throughput over latency.
- The fundamental hardware unit is the **warp** (32 threads executing in lockstep).
- CUDA organizes threads into a hierarchy: **thread → block → grid**.
- Memory is hierarchical: registers → shared memory → L2 → global DRAM.
- Every CUDA program follows: allocate → copy to GPU → kernel launch → sync → copy back → free.
- **Always check CUDA error codes.**

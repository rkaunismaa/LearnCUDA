# Chapter 03: The CUDA Memory Hierarchy

## 3.1 Why Memory Matters More Than Compute

On modern GPUs, arithmetic operations are extremely cheap. An RTX 4090 can perform over 80 **trillion** floating-point operations per second (TFLOPS). But it can only move ~1 TB/s of data from global DRAM.

Consider vector addition (`C = A + B`):
- 3 memory operations (read A, read B, write C) per 1 add
- At 1 TB/s and 4 bytes/float: 250 billion reads/writes per second
- At 1 add per 3 ops: ~83 billion adds per second = ~0.1% of peak compute!

**Most real CUDA programs are memory-bound.** Understanding the memory hierarchy is how you unlock GPU performance.

### The Roofline: Compute vs. Memory Bandwidth

```mermaid
flowchart LR
    subgraph COMPUTE["⚡ Compute Peak"]
        FP["82,600 GFLOPS FP32\nRTX 4090 can sustain\n82 FLOP per byte loaded"]
    end
    subgraph MEMORY["📦 Memory Peak"]
        BW["1,008 GB/s GDDR6X\n= 252 billion float\nreads per second"]
    end

    COMPUTE --> BALANCE["Arithmetic Intensity\n(FLOP / byte)"]
    MEMORY  --> BALANCE

    BALANCE -->|"kernel AI &lt; 82\n(vecAdd ≈ 0.08)"| MB["🐌 MEMORY-BOUND\nCompute sits idle\nwaiting for data\n→ optimize memory access"]
    BALANCE -->|"kernel AI &gt; 82\n(matmul ≈ 512)"| CB["🚀 COMPUTE-BOUND\nMemory keeps up\ncompute is bottleneck\n→ optimize arithmetic"]

    style COMPUTE fill:#1e8449,color:#fff,stroke:#196f3d
    style MEMORY  fill:#c0392b,color:#fff,stroke:#922b21
    style BALANCE fill:#7d3c98,color:#fff,stroke:#6c3483
    style MB fill:#7d3c98,color:#fff,stroke:#6c3483
    style CB fill:#1f618d,color:#fff,stroke:#154360
```

## 3.2 Memory Types Overview

```mermaid
graph TB
    subgraph CHIP["🔲 GPU Chip (RTX 4090)"]
        subgraph SM["⚙️  Streaming Multiprocessor  (128× on RTX 4090)"]
            REG["⚡ Registers\nPer-thread  |  ~1 cycle\n256 KB per SM\nCompiler-managed — fastest storage"]
            SMEM["🔥 Shared Memory / L1 Cache\nPer-block  |  ~5–30 cycles\n128 KB per SM\nProgrammer-managed scratchpad"]
            CC["📖 Constant Cache\nAll threads (read-only)  |  ~1 cycle broadcast\n~8 KB per SM"]
        end
        L2["💡 L2 Cache\nAll SMs  |  ~100–200 cycles  |  72 MB total"]
        GMEM["💾 Global Memory  (GDDR6X VRAM)\nAll threads  |  ~200–600 cycles  |  24 GB  |  ~1,008 GB/s"]
    end
    HOST["🖥️  System RAM (CPU DDR5)\nAll threads via cudaMemcpy  |  ~1,000+ cycles  |  ~64 GB/s via PCIe"]

    REG  -.->|"register spill →"| GMEM
    SM   --> L2
    L2   --> GMEM
    GMEM -.->|"cudaMemcpy"| HOST

    style CHIP fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style SM   fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style REG  fill:#c0392b,color:#fff,stroke:#922b21
    style SMEM fill:#d35400,color:#fff,stroke:#a04000
    style CC   fill:#7d3c98,color:#fff,stroke:#6c3483
    style L2   fill:#c8a000,color:#1a1a1a,stroke:#b7950b
    style GMEM fill:#1f618d,color:#fff,stroke:#154360
    style HOST fill:#4a235a,color:#d2b4de,stroke:#6c3483
```

## 3.3 Registers

- **Scope**: Per-thread (completely private)
- **Latency**: ~1 cycle
- **Size**: ~65,536 registers per SM (each 32-bit)
- **Management**: Automatic — compiler assigns variables to registers

Registers are the fastest storage. Local variables in a kernel live here.

```c
__global__ void kernel(float *data, int n)
{
    float temp = data[blockIdx.x];  // 'temp' lives in a register
    temp = temp * temp + 1.0f;      // register arithmetic — ~1 cycle
    data[blockIdx.x] = temp;
}
```

**Register pressure**: if a kernel uses too many registers, the compiler "spills" them to slow **local memory** (global DRAM with per-thread addressing). Avoid this.

## 3.4 Global Memory

- **Scope**: All threads, entire program lifetime
- **Latency**: ~200–600 cycles
- **Bandwidth**: ~1 TB/s (RTX 4090)
- **Declaration**: `cudaMalloc` / `cudaMallocManaged`

Global memory is large but slow. The critical optimization is **memory coalescing**.

### Memory Coalescing

When threads in a warp access global memory, the hardware combines (coalesces) adjacent accesses into fewer, wider transactions. A full warp of 32 threads ideally issues a single 128-byte transaction.

```diff
  ── Pattern 1: Coalesced — thread i reads data[i]  ──────────────────────────

+ Thread  0: data[ 0] → addr 0x0000  ┐
+ Thread  1: data[ 1] → addr 0x0004  │  contiguous addresses
+ Thread  2: data[ 2] → addr 0x0008  │
+ Thread  3: data[ 3] → addr 0x000C  │
+ ...                                 │
+ Thread 31: data[31] → addr 0x007C  ┘
+ → 1 × 128-byte transaction for the whole warp  |  100% efficiency ✓


  ── Pattern 2: Strided — thread i reads data[i * 2]  ────────────────────────

  Thread  0: data[ 0] → addr 0x0000  ┐
  Thread  1: data[ 2] → addr 0x0008  │  gaps between accesses
  Thread  2: data[ 4] → addr 0x0010  │  (every other element skipped)
  ...                                 │
  Thread 31: data[62] → addr 0x00F8  ┘
  → 2 × 128-byte transactions  |  50% wasted (unused elements loaded)


  ── Pattern 3: Random — thread i reads data[rand_idx[i]]  ───────────────────

- Thread  0: data[47]  → addr 0x00BC  ┐
- Thread  1: data[ 3]  → addr 0x000C  │  scattered across memory
- Thread  2: data[211] → addr 0x034C  │
- Thread  3: data[ 88] → addr 0x0160  │
- ...                                  │
- Thread 31: data[512] → addr 0x0800  ┘
- → up to 32 separate transactions  |  ~3% efficiency ✗
```

**Always arrange your data and access patterns to be coalesced.**

## 3.5 Shared Memory

- **Scope**: All threads within the same **block** (shared by the block)
- **Latency**: ~5–30 cycles (on-chip)
- **Size**: Up to 100 KB per SM (configurable)
- **Declaration**: `__shared__` keyword

Shared memory is a programmer-controlled cache. Use it when:
- Multiple threads in a block access the same data (reuse)
- You want to reorganize non-coalesced accesses into coalesced ones

```c
__global__ void sharedExample(float *input, float *output, int n)
{
    // Static shared memory declaration (size known at compile time)
    __shared__ float tile[256];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load from slow global memory into fast shared memory
    if (i < n)
        tile[threadIdx.x] = input[i];

    // Synchronize: ALL threads in the block must reach here before any continue
    __syncthreads();

    // Now threads can read each other's values from tile[]
    // e.g., thread i reads its neighbor's value
    if (i < n && threadIdx.x > 0)
        output[i] = (tile[threadIdx.x] + tile[threadIdx.x - 1]) / 2.0f;
}
```

### The `__syncthreads()` Barrier

`__syncthreads()` is a **barrier**: every thread in the block must reach it before any thread proceeds. This prevents data races where one thread reads before another has written.

```mermaid
sequenceDiagram
    participant T0  as Thread 0
    participant T1  as Thread 1
    participant TN  as Thread N-1
    participant SM  as Shared Memory tile[]

    Note over T0,TN: Phase 1 — Load from global → shared (all threads write)
    T0  ->> SM: tile[0]   = input[i0]
    T1  ->> SM: tile[1]   = input[i1]
    TN  ->> SM: tile[N-1] = input[iN]

    Note over T0,TN: ⚠️  __syncthreads() — barrier point
    T0  -->> T0:  wait (arrives early, blocks)
    TN  -->> TN:  wait (arrives late, all others waiting)
    Note over T0,TN: ALL threads arrived → barrier lifts ✓

    Note over T0,TN: Phase 2 — Read neighbors from shared (safe to read now)
    T0  ->> SM: read tile[1]   (neighbor)
    T1  ->> SM: read tile[0] and tile[2]
    TN  ->> SM: read tile[N-2]
    SM -->> T0: value
    SM -->> T1: values
    SM -->> TN: value

    Note over T0,TN: ✓ No data race — all writes visible to all readers
```

### Dynamic Shared Memory

When the size is unknown at compile time:

```c
// Kernel declaration
__global__ void myKernel(float *data, int n)
{
    extern __shared__ float smem[];  // size specified at launch
    // ...
}

// Launch with dynamic shared memory size as 3rd config parameter
myKernel<<<grid, block, sharedMemBytes>>>(data, n);
```

### Shared Memory Bank Conflicts

Shared memory is organized into 32 **banks** (for 4-byte words). If multiple threads in a warp access the same bank simultaneously, accesses are serialized — a **bank conflict**.

```diff
  Shared memory layout:  Bank = address % 32
  Banks:  [0] [1] [2] [3] ... [31] [0] [1] [2] ... (wraps every 32 words)
  Addrs:   0   1   2   3  ...  31   32  33  34  ...


  ── ✓ No Conflict: tile[threadIdx.x]  ─────────────────────────────────────

+ Thread  0 → tile[ 0] → Bank  0   ✓ unique bank
+ Thread  1 → tile[ 1] → Bank  1   ✓ unique bank
+ Thread  2 → tile[ 2] → Bank  2   ✓ unique bank
+ ...
+ Thread 31 → tile[31] → Bank 31   ✓ unique bank
+ → All 32 banks busy in parallel — 1 cycle ✓


  ── ✗ 2-Way Conflict: tile[threadIdx.x * 2]  ──────────────────────────────

- Thread  0 → tile[ 0] → Bank  0  ┐ CONFLICT: two threads
- Thread 16 → tile[32] → Bank  0  ┘ hit same bank → serialized
- Thread  1 → tile[ 2] → Bank  2  ┐ CONFLICT
- Thread 17 → tile[34] → Bank  2  ┘ same bank → serialized
- ...
- → 2 serialized passes — 2× slower ✗


  ── ✗ 32-Way Conflict: tile[0] (all write same addr, not broadcast)  ───────

- All 32 threads → tile[0] → Bank 0  → fully serialized: 32× slower ✗
  (Exception: reading the same address IS a free broadcast ✓)
```

See `02_shared_memory.cu` for a demonstration.

## 3.6 Constant Memory

- **Scope**: All threads (read-only from device, writable from host)
- **Latency**: ~1 cycle if all threads read the **same** address (broadcast)
- **Size**: 64 KB total (hardware-limited)
- **Declaration**: `__constant__` keyword (at file scope)

Perfect for read-only data used uniformly across all threads (filter kernels, lookup tables, physics constants).

```c
__constant__ float coeffs[16];  // at file scope

// Host sets it via:
cudaMemcpyToSymbol(coeffs, h_coeffs, 16 * sizeof(float));

// Device reads it normally:
__global__ void filter(float *data, int n)
{
    // All threads read coeffs[3] simultaneously → single broadcast, 0 latency
    float c = coeffs[3];
    // ...
}
```

### Constant Memory Broadcast vs. Serialization

```mermaid
graph TD
    HOST["🖥️  Host\ncudaMemcpyToSymbol(coeffs, ...)"]
    CMEM["📦 __constant__ memory  (64 KB)\ncoeffs[16] — read-only from device\nwritten once from host"]
    CACHE["🔍 Constant Cache  (~8 KB per SM)\nCaches recently-read constant values"]

    subgraph BROADCAST["✓ All 32 threads read coeffs[3] — same address"]
        TB0["Thread  0\ncoeffs[3]"]
        TB1["Thread  1\ncoeffs[3]"]
        TBD["..."]
        TB31["Thread 31\ncoeffs[3]"]
    end

    subgraph SERIAL["✗ All 32 threads read different addresses"]
        TS0["Thread  0\ncoeffs[0]"]
        TS1["Thread  1\ncoeffs[1]"]
        TSD["..."]
        TS31["Thread 31\ncoeffs[31]"]
    end

    HOST  -->|"writes once"| CMEM
    CMEM  --> CACHE
    CACHE -->|"✓ single broadcast\n~1 cycle for all 32"| BROADCAST
    CACHE -->|"✗ 32 serialized reads\n~32 cycles"| SERIAL

    style HOST      fill:#2c3e50,color:#ecf0f1,stroke:#1a252f
    style CMEM      fill:#7d3c98,color:#fff,stroke:#6c3483
    style CACHE     fill:#1f618d,color:#fff,stroke:#154360
    style BROADCAST fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style SERIAL    fill:#2a0d0d,color:#f1948a,stroke:#e74c3c
    style TB0       fill:#1e8449,color:#fff,stroke:#196f3d
    style TB1       fill:#1e8449,color:#fff,stroke:#196f3d
    style TBD       fill:#145a32,color:#a9dfbf,stroke:#1e8449
    style TB31      fill:#1e8449,color:#fff,stroke:#196f3d
    style TS0       fill:#c0392b,color:#fff,stroke:#922b21
    style TS1       fill:#c0392b,color:#fff,stroke:#922b21
    style TSD       fill:#922b21,color:#f1948a,stroke:#7b241c
    style TS31      fill:#c0392b,color:#fff,stroke:#922b21
```

**Key constraint**: if different threads access different constant memory addresses simultaneously, accesses are serialized — you lose the benefit. Use it only when all threads need the same value.

## 3.7 Local Memory

- **Scope**: Per-thread (private)
- **Physically**: Part of global DRAM (slow!)
- **Used**: When registers spill, or for large per-thread arrays

Local memory is a misnomer — it's not fast local memory, it's just private global memory. Avoid it.

```c
__global__ void careful(int n)
{
    int bigArray[100];  // DANGER: this may go to local memory!
    // ...
}
```

## 3.8 Memory Usage Summary

```mermaid
graph LR
    subgraph FAST["Fast — use as much as possible"]
        R["⚡ Registers\n1 cycle | 256 KB/SM\nThread-private\nCompiler assigns"]
        S["🔥 Shared Mem\n~30 cycles | 100 KB/SM\nBlock-shared\n__shared__ keyword"]
        C["📖 Constant Mem\n~1 cycle (broadcast)\n64 KB total\n__constant__ keyword"]
    end
    subgraph MID["Automatic caches — hardware managed"]
        L1["🟡 L1 Cache\n~30 cycles | 128 KB/SM\nShared with smem\nAutomatic"]
        L2["💡 L2 Cache\n~200 cycles | 72 MB\nAll SMs\nAutomatic"]
    end
    subgraph SLOW["Slow — minimize accesses"]
        G["💾 Global Mem\n~600 cycles | 24 GB\nAll threads\ncudaMalloc"]
        LM["🐌 Local Mem\n~600 cycles | per-thread\nRegister spill!\nAvoid"]
    end

    style FAST fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style MID  fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style SLOW fill:#2a0d0d,color:#f1948a,stroke:#e74c3c
    style R  fill:#c0392b,color:#fff,stroke:#922b21
    style S  fill:#d35400,color:#fff,stroke:#a04000
    style C  fill:#7d3c98,color:#fff,stroke:#6c3483
    style L1 fill:#c8a000,color:#1a1a1a,stroke:#b7950b
    style L2 fill:#c8a000,color:#1a1a1a,stroke:#b7950b
    style G  fill:#1f618d,color:#fff,stroke:#154360
    style LM fill:#922b21,color:#f1948a,stroke:#7b241c
```

| Memory | Scope | Latency | Size | Notes |
|--------|-------|---------|------|-------|
| Registers | Thread | 1 cycle | 256 KB/SM | Fast, limited |
| Shared | Block | ~30 cycles | 100 KB/SM | Programmer-managed L1 |
| L1 Cache | SM | ~30 cycles | 128 KB/SM | Automatic, shared with smem |
| Constant | All threads | ~1 cycle (broadcast) | 64 KB | Read-only, all-same-address |
| Texture | All threads | ~30 cycles | Backed by L2 | Spatial locality, read-only |
| L2 Cache | All SMs | ~200 cycles | 72 MB (4090) | Automatic |
| Global | All threads | ~600 cycles | 24 GB (4090) | Main VRAM |

## 3.9 Exercises

1. In `01_global_memory.cu`, change the access pattern to strided. Use `nvprof` or Nsight to observe the drop in effective bandwidth.
2. In `02_shared_memory.cu`, intentionally create a bank conflict by changing the access stride. Measure the performance difference.
3. Implement a 1D convolution using constant memory for the filter coefficients. Compare with using global memory for the filter.
4. What is the maximum number of 32-bit registers per thread? (Hint: each SM has 65,536 registers and max 2048 threads.)

## 3.10 Key Takeaways

- Most CUDA kernels are **memory-bound** — optimize memory access before worrying about arithmetic.
- **Coalesced global memory access** is the most impactful single optimization.
- **Shared memory** is a fast programmer-managed cache; use it to stage data for reuse.
- `__syncthreads()` is required after writes to shared memory before others read.
- **Constant memory** provides free broadcast for uniform read-only data.
- Avoid register spills to local memory (watch for large per-thread arrays).

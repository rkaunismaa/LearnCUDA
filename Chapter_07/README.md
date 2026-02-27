# Chapter 07: Profiling and Performance Analysis

## 7.1 The Performance Mindset

Before optimizing anything, **measure first**. Guessing where the bottleneck is will lead you astray. CUDA provides precise tools for understanding what's happening on the GPU.

```mermaid
flowchart LR
    M["📊 Measure\nCUDA events · nsys · ncu"]
    I["🔍 Identify Bottleneck\nMemory-bound or compute-bound?\nOccupancy? Divergence?"]
    O["🔧 Optimize the Right Thing\nWrong bottleneck → wasted effort"]
    V["✅ Verify\nDid it actually improve?\nNew bottleneck introduced?"]

    M --> I --> O --> V --> M

    style M fill:#1f618d,color:#fff,stroke:#154360
    style I fill:#7d3c98,color:#fff,stroke:#6c3483
    style O fill:#1e8449,color:#fff,stroke:#196f3d
    style V fill:#d35400,color:#fff,stroke:#a04000
```

The two key questions before optimizing:
1. **Is this kernel memory-bound or compute-bound?** (The roofline model answers this)
2. **Is the GPU being fully utilized?** (Occupancy and warp efficiency answer this)

## 7.2 CUDA Events for Precise Timing

CPU timing with `clock()` or `gettimeofday()` is inaccurate for GPU code because kernel launches are asynchronous. Use **CUDA events** — they are timestamped by the GPU itself.

```mermaid
sequenceDiagram
    participant CPU as 🖥️ Host (CPU)
    participant GPU as 🎮 GPU

    Note over CPU,GPU: ❌ Wrong — CPU timer measures only launch overhead
    CPU  ->> GPU:  kernel<<<grid,block>>>()   [returns immediately]
    CPU  ->> CPU:  stop = clock()  ← records ~microseconds, not kernel runtime ✗
    activate GPU
    Note over GPU: kernel still running for milliseconds...
    deactivate GPU

    Note over CPU,GPU: ✓ Correct — CUDA events timestamped by GPU hardware
    CPU  ->> GPU:  cudaEventRecord(start)      [inject timestamp into stream]
    CPU  ->> GPU:  kernel<<<grid,block>>>()
    CPU  ->> GPU:  cudaEventRecord(stop)       [inject timestamp after kernel]
    activate GPU
    Note over GPU: kernel executing...
    GPU -->> GPU:  stop event fires ✓
    deactivate GPU
    CPU  ->> GPU:  cudaEventSynchronize(stop)  [CPU blocks here until stop fires]
    GPU -->> CPU:  cudaEventElapsedTime(&ms, start, stop)  →  X.XXX ms ✓
```

```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);           // Inject timestamp into default stream
myKernel<<<grid, block>>>(...);   // Kernel runs asynchronously
cudaEventRecord(stop);

cudaEventSynchronize(stop);       // Block CPU until stop event fires

float ms;
cudaEventElapsedTime(&ms, start, stop);  // Time between events in milliseconds
printf("Kernel time: %.3f ms\n", ms);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

Key rules:
- `cudaEventRecord(e, stream)` places a timestamp in the stream's command queue
- Always call `cudaEventSynchronize(stop)` before reading elapsed time

## 7.3 The Roofline Model

The **roofline model** determines the theoretical peak performance of a kernel based on its **arithmetic intensity** (AI):

```
AI = Total FLOPs executed / Total bytes transferred to/from global memory
```

### The Roofline Shape

```
GFLOPS/s
  │
  │  82,600 ══════════════════════════════════════════ Compute ceiling (FP32)
  │         ╗
  │         ║  ← Compute-bound region (AI > 82 FLOP/byte)
  │         ║     Adding more compute helps; memory is not the limit
  │         ╝
  │   ╔═════╝
  │   ║   ← Memory-bound region (AI < 82 FLOP/byte)
  │   ║      Adding more compute does NOT help — you need less data movement
  │   ║
  └───╨─────────────────────────────────── FLOP/byte
   0.08  2   10   50  82(ridge)  200
```

### RTX 4090 Roofline — Where Do Kernels Land?

```mermaid
flowchart TD
    subgraph MEMBOUND["📦 Memory-Bound Region  (AI &lt; 82 FLOP/byte)\nPerformance = AI × 1,008 GB/s\nFix: reduce bytes moved, increase reuse"]
        VA["Vector Add\nAI ≈ 0.08 FLOP/byte\nCeiling: ~81 GFLOPS\nTypically achieves ~80 GFLOPS\n(near-optimal already ✓)"]
        TM["Tiled MatMul  T=16\nAI ≈ 2.0 FLOP/byte\nCeiling: ~2,016 GFLOPS\nTypically achieves ~500 GFLOPS\n(room to improve with larger tiles)"]
        CB["cuBLAS GEMM  (large)\nAI ≈ 50 FLOP/byte\nCeiling: ~50,000 GFLOPS\nAchieves ~20,000 GFLOPS\n(near ridge — well optimized)"]
    end

    RIDGE["⚡ Ridge Point: AI = 82 FLOP/byte\n= Peak Compute ÷ Peak Bandwidth\n= 82,600 GFLOPS ÷ 1,008 GB/s"]

    subgraph COMPBOUND["⚡ Compute-Bound Region  (AI &gt; 82 FLOP/byte)\nPerformance = 82,600 GFLOPS (FP32 ceiling)\nFix: reduce arithmetic, use Tensor Cores (330 TFLOPS FP16)"]
        TC["Tensor Core GEMM\nAI &gt;&gt; 82 FLOP/byte\nCeiling: 330,000 GFLOPS FP16\nAchieves ~280,000 GFLOPS"]
    end

    MEMBOUND --> RIDGE --> COMPBOUND

    style MEMBOUND  fill:#2a0d0d,color:#f1948a,stroke:#e74c3c
    style COMPBOUND fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style RIDGE     fill:#7d3c98,color:#fff,stroke:#6c3483
    style VA        fill:#c0392b,color:#fff,stroke:#922b21
    style TM        fill:#d35400,color:#fff,stroke:#a04000
    style CB        fill:#b7950b,color:#fff,stroke:#9a7d0a
    style TC        fill:#1e8449,color:#fff,stroke:#196f3d
```

## 7.4 SM Occupancy

**Occupancy** = (active warps per SM) / (maximum warps per SM)

Higher occupancy helps the GPU **hide memory latency** by switching between warps while one warp waits for a memory transaction.

### Little's Law Applied to GPUs

```mermaid
graph LR
    subgraph LAW["Little's Law:  Throughput = Concurrency / Latency"]
        EQ["To sustain 1 warp/cycle throughput\nwith 300-cycle memory latency:\n→ need 300 warps in flight simultaneously\nRTX 4090: 64 warps/SM × 128 SMs = 8,192 max concurrent warps"]
    end

    subgraph LOWOC["Low Occupancy — stalls visible"]
        W0L["Warp 0\nloads memory\n(300 cycle wait)"]
        STALL["SM STALLS\n~270 idle cycles\nwaiting for data ✗"]
        W0L --> STALL
    end

    subgraph HIGHOC["High Occupancy — latency hidden"]
        W0H["Warp 0\nloads memory"]
        W1H["Warp 1 runs\n(cycles 1-32)"]
        W2H["Warp 2 runs\n(cycles 33-64)"]
        WNH["Warps 3-9 run\n(cycles 65-288)"]
        W0R["Warp 0 data ready\nresumes at cycle ~300\nno visible stall ✓"]
        W0H --> W1H --> W2H --> WNH --> W0R
    end

    style LAW    fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style LOWOC  fill:#2a0d0d,color:#f1948a,stroke:#e74c3c
    style HIGHOC fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style EQ     fill:#1f618d,color:#fff,stroke:#154360
    style W0L    fill:#c0392b,color:#fff,stroke:#922b21
    style STALL  fill:#7b241c,color:#f1948a,stroke:#c0392b
    style W0H    fill:#1e8449,color:#fff,stroke:#196f3d
    style W1H    fill:#1e8449,color:#fff,stroke:#196f3d
    style W2H    fill:#1e8449,color:#fff,stroke:#196f3d
    style WNH    fill:#1e8449,color:#fff,stroke:#196f3d
    style W0R    fill:#145a32,color:#a9dfbf,stroke:#1e8449
```

### The Three Occupancy Limiters

```mermaid
graph TD
    SM["⚙️ SM Resources  (RTX 4090 per SM)\nMax warps: 64  |  Max threads: 2048  |  Max blocks: 32"]

    SM --> REG
    SM --> SMEM
    SM --> BLK

    subgraph REG["📦 Registers  (65,536 per SM)"]
        R1["32 regs/thread:\n65536 ÷ (32×32) = 64 warps → 100% ✓"]
        R2["64 regs/thread:\n65536 ÷ (64×32) = 32 warps → 50% ⚠️"]
        R3["128 regs/thread:\n65536 ÷ (128×32) = 16 warps → 25% ✗"]
    end

    subgraph SMEM["🔥 Shared Memory  (128 KB per SM)"]
        S1["8 KB/block, 256 threads:\n128÷8=16 blocks, 4096 threads → 100% ✓"]
        S2["32 KB/block, 256 threads:\n128÷32=4 blocks, 1024 threads → 50% ⚠️"]
        S3["64 KB/block, 256 threads:\n128÷64=2 blocks, 512 threads → 25% ✗"]
    end

    subgraph BLK["🔷 Block Size  (max 1024 threads/block)"]
        B1["blockDim=256: 2048÷256=8 blocks → 100% ✓"]
        B2["blockDim=512: 2048÷512=4 blocks → 100% ✓"]
        B3["blockDim=64:  2048÷64=32 blocks → 100% ✓\n(but may underutilize warp schedulers)"]
    end

    style SM   fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style REG  fill:#1f618d,color:#aed6f1,stroke:#2980b9
    style SMEM fill:#7d3c98,color:#d2b4de,stroke:#8e44ad
    style BLK  fill:#1e8449,color:#a9dfbf,stroke:#27ae60
    style R1   fill:#145a32,color:#a9dfbf,stroke:#1e8449
    style R2   fill:#c8a000,color:#1a1a1a,stroke:#b7950b
    style R3   fill:#922b21,color:#f1948a,stroke:#c0392b
    style S1   fill:#145a32,color:#a9dfbf,stroke:#1e8449
    style S2   fill:#c8a000,color:#1a1a1a,stroke:#b7950b
    style S3   fill:#922b21,color:#f1948a,stroke:#c0392b
    style B1   fill:#145a32,color:#a9dfbf,stroke:#1e8449
    style B2   fill:#145a32,color:#a9dfbf,stroke:#1e8449
    style B3   fill:#c8a000,color:#1a1a1a,stroke:#b7950b
```

```c
// Query occupancy for a given kernel and block size
int activeBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &activeBlocks, myKernel, blockSize, sharedMemBytes);

cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
float occupancy = (float)(activeBlocks * blockSize)
                / prop.maxThreadsPerMultiProcessor;
printf("Occupancy: %.0f%%\n", occupancy * 100);
```

## 7.5 Nsight Systems and Nsight Compute

```mermaid
flowchart TD
    Q{"Which profiling\ntool do I need?"}

    Q -->|"Big picture:\nIs stream overlap working?\nWhere does GPU sit idle?\nWhich kernel takes longest?"| NSYS

    Q -->|"Deep dive:\nWhy is this kernel slow?\nMemory throughput %?\nOccupancy? Cache hit rates?"| NCU

    subgraph NSYS["🔭 nsys  (Nsight Systems) — System-Level Timeline"]
        NS1["nsys profile --trace=cuda,nvtx -o report ./program"]
        NS2["Shows: CPU activity · GPU kernels · memcpy · streams"]
        NS3["Open: nsys-ui report.nsys-rep"]
        NS1 --> NS2 --> NS3
    end

    subgraph NCU["🔬 ncu  (Nsight Compute) — Per-Kernel Hardware Metrics"]
        NC1["ncu --set full -o report ./program"]
        NC2["Shows: memory throughput · occupancy · warp efficiency\nL1/L2 hit rates · stall reasons · instruction mix"]
        NC3["Open: ncu-ui report.ncu-rep"]
        NC1 --> NC2 --> NC3
    end

    style Q    fill:#7d3c98,color:#fff,stroke:#6c3483
    style NSYS fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style NCU  fill:#0d1b2a,color:#aed6f1,stroke:#2980b9
    style NS1  fill:#1e8449,color:#fff,stroke:#196f3d
    style NS2  fill:#145a32,color:#a9dfbf,stroke:#1e8449
    style NS3  fill:#1e8449,color:#fff,stroke:#196f3d
    style NC1  fill:#1f618d,color:#fff,stroke:#154360
    style NC2  fill:#154360,color:#aed6f1,stroke:#1f618d
    style NC3  fill:#1f618d,color:#fff,stroke:#154360
```

### Key Nsight Compute Metrics

| Metric | What it tells you | Good value |
|--------|------------------|------------|
| `Memory Throughput %` | How close to peak bandwidth | > 70% |
| `SM Active Cycles %` | SM utilization | > 80% |
| `Warp Efficiency %` | Non-divergent warp fraction | > 90% |
| `L1 Hit Rate %` | L1 cache effectiveness | > 80% |
| `L2 Hit Rate %` | L2 cache effectiveness | > 60% |
| `Achieved Occupancy` | Actual warp occupancy at runtime | > 50% |

## 7.6 Quick Profiling Commands

```bash
# Time all kernels, no GUI needed
ncu --csv --print-summary per-kernel ./my_program

# Check memory throughput for specific kernel
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
dram__bytes_read.sum,dram__bytes_write.sum ./my_program

# System-level view
nsys profile -d 5 ./my_program

# Profile with NVTX code annotations
# Add to code: nvtxRangePush("MySection"); ... nvtxRangePop();
nsys profile --trace=cuda,nvtx ./my_program
```

### Reading the Results: Diagnosis Flowchart

```mermaid
flowchart TD
    START["🔬 Kernel is slower than expected"]

    MEM{"ncu: Memory\nThroughput % high?"}
    OCC{"ncu: Achieved\nOccupancy low?"}
    DIV{"ncu: Warp\nEfficiency low?"}
    CACHE{"ncu: L1/L2\nHit Rate low?"}
    COMP{"ncu: SM Active %\nhigh but BW low?"}

    START --> MEM

    MEM -->|"Yes > 80%\nmemory-bound"| COALFIX["Fix: improve coalescing\nuse shared memory tiling\nreduce global memory ops"]
    MEM -->|"No < 50%"| OCC

    OCC -->|"Yes < 50%"| OCCFIX["Fix: reduce registers\nreduce shared mem\nadjust block size"]
    OCC -->|"No"| DIV

    DIV -->|"Yes < 80%"| DIVFIX["Fix: remove branch divergence\nrestructure data layout\nuse warp-uniform conditions"]
    DIV -->|"No"| CACHE

    CACHE -->|"Yes < 50%"| CACHEFIX["Fix: improve spatial locality\nuse __ldg() for read-only\nuse shared memory for reuse"]
    CACHE -->|"No"| COMP

    COMP -->|"Yes"| COMPFIX["Compute-bound!\nFix: reduce arithmetic\nuse Tensor Cores\nuse fast math (--use_fast_math)"]

    style START   fill:#c0392b,color:#fff,stroke:#922b21
    style MEM     fill:#7d3c98,color:#fff,stroke:#6c3483
    style OCC     fill:#7d3c98,color:#fff,stroke:#6c3483
    style DIV     fill:#7d3c98,color:#fff,stroke:#6c3483
    style CACHE   fill:#7d3c98,color:#fff,stroke:#6c3483
    style COMP    fill:#7d3c98,color:#fff,stroke:#6c3483
    style COALFIX fill:#1e8449,color:#fff,stroke:#196f3d
    style OCCFIX  fill:#1e8449,color:#fff,stroke:#196f3d
    style DIVFIX  fill:#1e8449,color:#fff,stroke:#196f3d
    style CACHEFIX fill:#1e8449,color:#fff,stroke:#196f3d
    style COMPFIX fill:#1e8449,color:#fff,stroke:#196f3d
```

## 7.7 Exercises

1. Run `01_cuda_events.cu`. Implement a function that measures the time of 3 different kernel configurations and reports bandwidth for each.
2. In `02_occupancy.cu`, add the `__launch_bounds__(256, 4)` qualifier to a kernel and observe how it changes register usage and occupancy.
3. Calculate the arithmetic intensity of tiled matrix multiply from Chapter 04. Based on the roofline model, is it memory-bound or compute-bound?
4. Run Nsight Compute on `01_vector_add` from Chapter 02: `ncu ./vec_add`. What is the achieved memory bandwidth %?
5. Add NVTX annotations to the Chapter 06 async pipeline. Run nsys to verify the overlap is visible in the timeline.

## 7.8 Key Takeaways

- Always use **CUDA events** (not CPU timers) to time GPU kernels — kernels are asynchronous.
- The **roofline model** tells you if you're memory-bound or compute-bound — optimize the right bottleneck.
- **Occupancy** helps hide memory latency via warp switching (Little's Law). 50%+ is usually sufficient.
- Three limiters of occupancy: **registers**, **shared memory**, **block size**.
- **Nsight Systems** for pipeline-level view (stream overlap, CPU/GPU timeline).
- **Nsight Compute** for kernel-level metrics (memory throughput, occupancy, stall reasons).
- `cudaOccupancyMaxPotentialBlockSize` automates finding the optimal block size.

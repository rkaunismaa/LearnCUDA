# Chapter 07: Profiling and Performance Analysis

## 7.1 The Performance Mindset

Before optimizing anything, **measure first**. Guessing where the bottleneck is will lead you astray. CUDA provides precise tools for understanding what's happening on the GPU.

The two key questions to answer before optimizing:
1. **Is this kernel memory-bound or compute-bound?** (The roofline model answers this)
2. **Is the GPU being fully utilized?** (Occupancy and warp efficiency answer this)

## 7.2 CUDA Events for Precise Timing

CPU timing with `clock()` or `gettimeofday()` is inaccurate for GPU code because kernel launches are asynchronous. Use **CUDA events** instead — they are timestamped by the GPU itself.

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
- The default (no stream arg) is `cudaStreamLegacy` (stream 0)
- Always call `cudaEventSynchronize(stop)` before reading elapsed time

## 7.3 The Roofline Model

The **roofline model** determines the theoretical peak performance of a kernel based on its **arithmetic intensity** (AI):

```
AI = Total FLOPs / Total bytes moved from/to global memory
```

```
         Peak Compute (TFLOPS)
         ┤                           ═══════════════
         │                  ════════╝
         │           ══════╝
GFLOPS/s │     ══════╝          ← Compute-bound region
         │ ════╝
         ┤╝ ← Memory-bound region
         └──────────────────────────────────
                   Arithmetic Intensity (FLOP/byte)
```

- **Below the ridge point**: memory-bound. Adding more FLOPs won't help; you need to move less data.
- **Above the ridge point**: compute-bound. You need more ALU throughput.

**RTX 4090 roofline:**
- Peak memory bandwidth: ~1008 GB/s
- Peak FP32 throughput: ~82.6 TFLOPS
- Ridge point: 82,600 / 1,008 ≈ **82 FLOP/byte**

Examples:
- Vector add: 1 FLOP / 12 bytes ≈ 0.08 FLOP/byte → deeply memory-bound
- Tiled matmul (TILE=16): ~2 FLOP/byte → memory-bound
- cuBLAS GEMM (large): ~50+ FLOP/byte → approaching compute-bound

## 7.4 SM Occupancy

**Occupancy** = (active warps per SM) / (maximum warps per SM)

Higher occupancy helps the GPU hide memory latency by switching between warps while one warp waits for a memory transaction (this is **Little's Law** applied to GPU execution).

**Little's Law for GPUs**: `Throughput = Concurrency / Latency`
- If memory latency = 300 cycles and we need 1 FLOP/cycle throughput, we need 300 in-flight warps.
- Each SM can hold up to 64 warps (RTX 4090). At 100% occupancy = 64 warps × 128 SMs = 8192 concurrent warps.

Occupancy is limited by three resources:
1. **Registers per thread**: RTX 4090 has 65,536 registers/SM. If each thread uses 64 registers, max warps = 65536/(64×32) = 32 warps = 50% occupancy.
2. **Shared memory per block**: Total 100KB/SM shared with L1. Large shared memory reduces blocks per SM.
3. **Block size**: Blocks must fit within SM resource limits.

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

NVIDIA provides two profiling tools:

### Nsight Systems (nsys) — System-level timeline
Shows the big picture: CPU activity, GPU kernels, memory copies, streams.

```bash
nsys profile --trace=cuda,nvtx -o report ./my_program
nsys-ui report.nsys-rep   # Open GUI
```

Use this to:
- See timeline of kernels and memory copies
- Find CPU-GPU synchronization bottlenecks
- Verify stream overlap is actually happening

### Nsight Compute (ncu) — Kernel-level deep dive
Shows detailed hardware metrics for individual kernels.

```bash
ncu --set full -o report ./my_program
ncu-ui report.ncu-rep   # Open GUI
# Or print to console:
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed my_program
```

Key metrics to check:
| Metric | What it tells you |
|--------|------------------|
| `Memory Throughput %` | How close to peak bandwidth you are |
| `SM Active Cycles %` | SM utilization |
| `Warp Efficiency %` | How often non-divergent warps execute |
| `L1 Hit Rate %` | Cache effectiveness |
| `L2 Hit Rate %` | L2 cache effectiveness |
| `Achieved Occupancy` | Actual warp occupancy at runtime |

## 7.6 Quick Profiling Commands

```bash
# Time all kernels (no GUI needed)
ncu --csv --print-summary per-kernel ./my_program

# Check memory throughput for specific kernel
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
dram__bytes_read.sum,dram__bytes_write.sum ./my_program

# System-level view with 5-second profiling
nsys profile -d 5 ./my_program

# Profile with NVTX ranges (code annotations)
# Add to your code: nvtxRangePush("MySection"); ... nvtxRangePop();
nsys profile --trace=cuda,nvtx ./my_program
```

## 7.7 Exercises

1. Run `01_cuda_events.cu`. Implement a function that measures the time of 3 different kernel configurations and reports bandwidth for each.
2. In `02_occupancy.cu`, add the `__launch_bounds__(256, 4)` qualifier to a kernel and observe how it changes register usage and occupancy.
3. Calculate the arithmetic intensity of tiled matrix multiply from Chapter 04. Based on the roofline model, is it memory-bound or compute-bound?
4. Run Nsight Compute on `01_vector_add` from Chapter 02: `ncu ./vec_add`. What is the achieved memory bandwidth %?
5. Add NVTX annotations to the Chapter 06 async pipeline. Run nsys to verify the overlap is visible in the timeline.

## 7.8 Key Takeaways

- Always use **CUDA events** (not CPU timers) to time GPU kernels.
- The **roofline model** tells you if you're memory-bound or compute-bound — optimize the right bottleneck.
- **Occupancy** helps hide memory latency; 50%+ is usually sufficient.
- Three limiters of occupancy: registers, shared memory, block size.
- **Nsight Systems** for pipeline-level view; **Nsight Compute** for kernel-level metrics.
- `cudaOccupancyMaxPotentialBlockSize` automates finding the optimal block size.

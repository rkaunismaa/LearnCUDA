# Chapter 08: Unified Memory and Pinned Memory

## 8.1 The Memory Transfer Problem

Every CUDA program we've written so far has this boilerplate:
```c
cudaMalloc → cudaMemcpy (H2D) → kernel → cudaMemcpy (D2H) → cudaFree
```

This explicit management is efficient but tedious. CUDA offers two alternatives:
- **Pinned memory**: same model, but H2D/D2H transfers are faster
- **Unified memory**: single pointer accessible from both CPU and GPU — no explicit copies

## 8.2 Pageable vs Pinned Host Memory

### Pageable Memory (normal `malloc`)
The OS can swap pageable pages to disk at any time. Before DMA (Direct Memory Access) can transfer data to the GPU, the CUDA driver must:
1. Allocate a temporary pinned staging buffer
2. Copy from pageable → pinned (CPU memcpy)
3. DMA from pinned → GPU

This double-copy costs bandwidth.

### Pinned (Page-Locked) Memory (`cudaMallocHost`)
Page-locked memory cannot be swapped. The GPU's DMA engine can transfer directly:
1. DMA from pinned → GPU

No staging copy needed. **Result**: ~15-30% faster H2D/D2H transfers.

```c
// Allocate pinned memory
float *h_pinned;
cudaMallocHost(&h_pinned, bytes);      // Page-locked

// Use exactly like malloc'd memory
for (int i = 0; i < n; i++) h_pinned[i] = (float)i;

// Transfer is faster
cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice);

// Required for async transfers (cudaMemcpyAsync)
cudaMemcpyAsync(d_data, h_pinned, bytes, cudaMemcpyHostToDevice, stream);

// Must free with cudaFreeHost (NOT free()!)
cudaFreeHost(h_pinned);
```

**Warning**: Too much pinned memory degrades system performance (less physical RAM available for OS paging). Use it selectively for large, frequently-transferred buffers.

## 8.3 Unified Memory (UM)

Unified Memory (introduced in CUDA 6, significantly improved in CUDA 8+) provides a **single pointer** that both CPU and GPU can dereference:

```c
float *data;
cudaMallocManaged(&data, bytes);

// CPU can write directly
for (int i = 0; i < n; i++) data[i] = (float)i;

// GPU can read/write directly in kernel
myKernel<<<grid, block>>>(data, n);
cudaDeviceSynchronize();

// CPU can read back directly (no cudaMemcpy needed!)
printf("data[0] = %f\n", data[0]);

cudaFree(data);  // Note: cudaFree, not cudaMallocHost/Host
```

### How UM Works Under the Hood

Unified Memory uses **page migration**. Each 4KB page of UM can live in either CPU RAM or GPU VRAM at any given time.

```
Initial state: all pages in CPU RAM (after cudaMallocManaged)

GPU kernel accesses page X (not in GPU memory):
  → Page fault on GPU
  → Driver migrates page X from CPU → GPU VRAM
  → Kernel continues

CPU accesses page Y (currently in GPU VRAM):
  → Page fault on CPU
  → Driver migrates page Y from GPU → CPU RAM
  → CPU continues
```

**Page faults are slow** (microseconds each). For large arrays, millions of page faults can dominate runtime.

## 8.4 Prefetching and Memory Advisories

### cudaMemPrefetchAsync
Tell the driver to move pages proactively before you need them — no page faults:

```c
int device = 0;
cudaGetDevice(&device);

// Prefetch entire array to GPU before kernel
cudaMemPrefetchAsync(data, bytes, device, stream);
myKernel<<<grid, block, 0, stream>>>(data, n);

// Prefetch back to CPU before reading
cudaMemPrefetchAsync(data, bytes, cudaCpuDeviceId, stream);
cudaStreamSynchronize(stream);
printf("data[0] = %f\n", data[0]);  // No page fault!
```

### cudaMemAdvise
Hints to the driver about access patterns:

```c
// Data primarily lives on GPU (avoid migration to CPU)
cudaMemAdvise(data, bytes, cudaMemAdviseSetPreferredLocation, device);

// Data is read by GPU but owned by CPU (map rather than migrate)
cudaMemAdvise(data, bytes, cudaMemAdviseSetReadMostly, device);

// CPU also accesses — set up direct mapping
cudaMemAdvise(data, bytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
```

## 8.5 Zero-Copy Memory

An alternative: map host memory directly into the GPU's address space. No migration — GPU reads over PCIe on every access.

```c
float *h_data, *d_data;
cudaHostAlloc(&h_data, bytes, cudaHostAllocMapped);  // Pinned + mapped
cudaHostGetDevicePointer(&d_data, h_data, 0);        // Get GPU-side pointer

// Kernel accesses d_data → goes over PCIe (~16 GB/s vs 1 TB/s VRAM)
myKernel<<<grid, block>>>(d_data, n);
```

Zero-copy is useful when:
- Data is accessed only once (no reuse benefit from migration)
- Data is too large to fit in GPU VRAM
- CPU and GPU access the same data roughly equally

## 8.6 When to Use Each

| Approach | Use When |
|----------|---------|
| Regular malloc + cudaMemcpy | Maximum performance, large repeated transfers |
| Pinned (cudaMallocHost) | Same as above but need faster transfers or async copies |
| Unified Memory + prefetch | Simpler code, data access pattern known ahead of time |
| Unified Memory (no prefetch) | Prototyping, irregular access patterns |
| Zero-copy | Data > VRAM, accessed once, or strong CPU-GPU sharing |

## 8.7 Exercises

1. In `01_pinned_memory.cu`, add a 4 GB transfer test. At what size does the bandwidth difference between pageable and pinned stabilize?
2. In `02_unified_memory.cu`, comment out the prefetch calls. Run and observe the slowdown from page faults. How much slower is it?
3. Try running a kernel on unified memory **without** `cudaDeviceSynchronize` before CPU access. What happens? Why?
4. Implement a benchmark using `cudaMemAdvise(data, bytes, cudaMemAdviseSetReadMostly, device)` for a read-only data array. Does it improve performance?

## 8.8 Key Takeaways

- **Pinned memory** eliminates the staging copy for H2D/D2H transfers, giving 15-30% faster transfers.
- **Pinned memory is required** for `cudaMemcpyAsync` to be truly asynchronous.
- **Unified memory** uses a single pointer for CPU+GPU but incurs page fault costs on first access.
- Use **cudaMemPrefetchAsync** to eliminate page faults by migrating data proactively.
- **Zero-copy** maps CPU memory into GPU space — useful for large data or infrequent access.
- Don't over-use pinned memory — it reduces available physical RAM for the OS.

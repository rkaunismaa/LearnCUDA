# Chapter 08: Unified Memory and Pinned Memory

## 8.1 The Memory Transfer Problem

Every CUDA program we've written so far has this boilerplate:
```c
cudaMalloc → cudaMemcpy (H2D) → kernel → cudaMemcpy (D2H) → cudaFree
```

This explicit management is efficient but tedious. CUDA offers two alternatives:

```mermaid
graph TD
    subgraph EXPLICIT["Explicit Management  (Chapters 01–07)"]
        E["cudaMalloc + cudaMemcpy\nFull programmer control\nFastest when done right"]
    end
    subgraph PINNED["Pinned Memory  (Section 8.2)"]
        P["cudaMallocHost\nSame explicit model\nbut transfers are 15–30% faster\n+ enables true async copies"]
    end
    subgraph UNIFIED["Unified Memory  (Section 8.3)"]
        U["cudaMallocManaged\nSingle pointer — CPU and GPU share it\nNo explicit cudaMemcpy needed\nDriver migrates pages on demand"]
    end
    subgraph ZEROCOPY["Zero-Copy  (Section 8.5)"]
        Z["cudaHostAlloc(mapped)\nGPU reads CPU RAM over PCIe\nNo migration, no VRAM used\nSlower access (~16 GB/s)"]
    end

    EXPLICIT -->|"add speed"| PINNED
    EXPLICIT -->|"reduce code"| UNIFIED
    EXPLICIT -->|"data > VRAM"| ZEROCOPY

    style EXPLICIT  fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style PINNED    fill:#1f618d,color:#fff,stroke:#154360
    style UNIFIED   fill:#1e8449,color:#fff,stroke:#196f3d
    style ZEROCOPY  fill:#7d3c98,color:#fff,stroke:#6c3483
    style E fill:#154360,color:#aed6f1,stroke:#1f618d
    style P fill:#154360,color:#aed6f1,stroke:#1f618d
    style U fill:#196f3d,color:#a9dfbf,stroke:#1e8449
    style Z fill:#6c3483,color:#d2b4de,stroke:#7d3c98
```

## 8.2 Pageable vs Pinned Host Memory

### Transfer Path Comparison

```mermaid
graph LR
    subgraph PAG["❌ Pageable Memory  (malloc)  —  2 copies"]
        PM["Pageable RAM\n(OS may swap to disk)"]
        SB["Temporary Pinned\nStaging Buffer\nCUDA driver allocates this"]
        GV1["GPU VRAM"]
        PM -->|"① CPU memcpy\nslow + blocks CPU"| SB
        SB -->|"② PCIe DMA\n~14–18 GB/s"| GV1
    end

    subgraph PIN["✓ Pinned Memory  (cudaMallocHost)  —  1 copy"]
        PH["Pinned (page-locked) RAM\nOS cannot swap — always resident"]
        GV2["GPU VRAM"]
        PH -->|"① Direct PCIe DMA\n~20–26 GB/s\n15–30% faster ✓"| GV2
    end

    style PAG fill:#2a0d0d,color:#f1948a,stroke:#e74c3c
    style PIN fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style PM  fill:#c0392b,color:#fff,stroke:#922b21
    style SB  fill:#922b21,color:#f1948a,stroke:#7b241c
    style GV1 fill:#1f618d,color:#fff,stroke:#154360
    style PH  fill:#1e8449,color:#fff,stroke:#196f3d
    style GV2 fill:#1f618d,color:#fff,stroke:#154360
```

```diff
  Pageable (malloc) path:

- Step 1: CUDA driver allocates a temporary pinned staging buffer  (hidden cost)
- Step 2: CPU copies pageable RAM → staging buffer               (wastes bandwidth)
- Step 3: DMA engine transfers staging buffer → GPU VRAM
- Total:  ~14–18 GB/s effective H2D bandwidth  ✗

  Pinned (cudaMallocHost) path:

+ Step 1: DMA engine transfers pinned RAM → GPU VRAM directly
+ Total:  ~20–26 GB/s effective H2D bandwidth  ✓  (full PCIe Gen4 ×16)
+ Bonus:  cudaMemcpyAsync is truly non-blocking (Chapter 06 streams) ✓
```

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

Unified Memory provides a **single pointer** that both CPU and GPU can dereference — no explicit `cudaMemcpy` needed.

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

cudaFree(data);
```

### How Unified Memory Works Under the Hood

UM uses **page migration** — each 4 KB page lives in either CPU RAM or GPU VRAM at any one time, migrating on demand:

```mermaid
sequenceDiagram
    participant CPU as 🖥️ CPU RAM
    participant DRV as CUDA Driver
    participant GPU as 🎮 GPU VRAM

    Note over CPU,GPU: After cudaMallocManaged — all pages resident in CPU RAM

    CPU  ->> CPU:  data[i] = value  ✓  (CPU writes — no fault, page already here)

    Note over CPU,GPU: kernel<<<grid,block>>>(data, n) — GPU starts accessing pages

    GPU  ->> DRV:  ⚠️ Page fault! Page for data[0..4095] not in VRAM
    DRV  ->> CPU:  invalidate CPU mapping, migrate page
    CPU -->> GPU:  page transferred via PCIe  (~microseconds)
    GPU  ->> GPU:  kernel continues on migrated page

    GPU  ->> DRV:  ⚠️ Page fault! Next page not in VRAM
    Note over GPU,DRV: Millions of faults for large arrays → very slow ✗

    Note over CPU,GPU: cudaDeviceSynchronize() — kernel done, pages now in VRAM

    CPU  ->> DRV:  ⚠️ Page fault! data[0] accessed but page now in VRAM
    DRV  ->> GPU:  migrate page back to CPU RAM
    GPU -->> CPU:  page returned via PCIe
    CPU  ->> CPU:  printf(data[0])  ✓  (slow first access, fast thereafter)
```

### Page Fault Cost

```diff
  Unified Memory WITHOUT prefetch — page faults on every new page:

- GPU kernel starts → page fault (data[0..4095] not in VRAM)
-   → driver migrates 4 KB page: ~10–50 µs overhead
- GPU continues → page fault (data[4096..8191] not in VRAM)
-   → driver migrates 4 KB page: ~10–50 µs overhead
- GPU continues → page fault (data[8192..12287]) ...
-   → repeat for every 4 KB page in the array
-
- For 256 MB array = 65,536 pages × ~20 µs = ~1.3 seconds of fault overhead ✗
- (cudaMemcpy would copy 256 MB in ~10 ms — 130× faster!) ✗

  Unified Memory WITH cudaMemPrefetchAsync — zero faults during kernel:

+ cudaMemPrefetchAsync(data, bytes, device, stream)
+   → Bulk DMA transfer of ALL pages to GPU VRAM
+   → Same path as cudaMemcpy: ~10 ms for 256 MB ✓
+ kernel runs with all data already in VRAM → zero page faults ✓
+ Near-identical performance to explicit cudaMemcpy ✓
```

## 8.4 Prefetching and Memory Advisories

### cudaMemPrefetchAsync

```mermaid
graph LR
    subgraph NOPF["Without Prefetch — demand migration"]
        KN["Kernel accesses\npage N"]
        FN["Page fault\n~20 µs per page"]
        MN["Page migrates\nfrom CPU → GPU"]
        KN --> FN --> MN --> KN
    end

    subgraph WITHPF["With cudaMemPrefetchAsync — proactive migration"]
        PF["cudaMemPrefetchAsync\n(data, bytes, device, stream)\nBulk DMA: all pages at once"]
        READY["All pages in VRAM\nbefore kernel starts"]
        KP["Kernel runs\nat full bandwidth\nzero page faults ✓"]
        PF --> READY --> KP
    end

    style NOPF  fill:#2a0d0d,color:#f1948a,stroke:#e74c3c
    style WITHPF fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style KN    fill:#c0392b,color:#fff,stroke:#922b21
    style FN    fill:#7b241c,color:#f1948a,stroke:#c0392b
    style MN    fill:#922b21,color:#f1948a,stroke:#7b241c
    style PF    fill:#1e8449,color:#fff,stroke:#196f3d
    style READY fill:#1e8449,color:#fff,stroke:#196f3d
    style KP    fill:#145a32,color:#a9dfbf,stroke:#1e8449
```

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

```mermaid
graph LR
    subgraph HOST["🖥️ Host (CPU RAM)"]
        HMEM["h_data[]\npinned + mapped\ncudaHostAlloc(cudaHostAllocMapped)\nalways in CPU RAM"]
    end

    subgraph PCIE["⚡ PCIe Bus  (~16 GB/s peak)"]
        BUS["Every GPU read/write\ncrosses here\nno caching in VRAM"]
    end

    subgraph DEVICE["🎮 GPU"]
        DPTR["d_data pointer\nmaps to h_data\n(no data in VRAM)"]
        KERN["kernel accesses d_data[i]\n→ PCIe fetch per cache line\n→ ~16 GB/s vs 1,008 GB/s ✗\n(but no VRAM used ✓)"]
    end

    HMEM   -->|"cudaHostGetDevicePointer\n(get GPU-visible address)"| DPTR
    KERN   -->|"memory load"| BUS
    BUS    -->|"fetch from CPU RAM"| HMEM

    style HOST   fill:#1f618d,color:#fff,stroke:#154360
    style PCIE   fill:#7d3c98,color:#fff,stroke:#6c3483
    style DEVICE fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style HMEM   fill:#154360,color:#aed6f1,stroke:#1f618d
    style BUS    fill:#6c3483,color:#d2b4de,stroke:#7d3c98
    style DPTR   fill:#196f3d,color:#a9dfbf,stroke:#1e8449
    style KERN   fill:#145a32,color:#a9dfbf,stroke:#1e8449
```

```c
float *h_data, *d_data;
cudaHostAlloc(&h_data, bytes, cudaHostAllocMapped);  // Pinned + mapped
cudaHostGetDevicePointer(&d_data, h_data, 0);        // Get GPU-side pointer

// Kernel accesses d_data → reads over PCIe (~16 GB/s vs 1 TB/s VRAM)
myKernel<<<grid, block>>>(d_data, n);
```

Zero-copy is useful when:
- Data is accessed only once (no reuse benefit from migration)
- Data is too large to fit in GPU VRAM
- CPU and GPU access the same data roughly equally

## 8.6 Bandwidth at a Glance

```mermaid
graph LR
    subgraph BW["Effective Bandwidth Comparison  (RTX 4090, PCIe Gen4 ×16)"]
        G["🔴 Global Memory (kernel access)\n~1,008 GB/s\nFastest — data already in VRAM"]
        PIN["🟠 Pinned + cudaMemcpy\n~20–26 GB/s\nFull PCIe bandwidth\nDirect DMA"]
        UMP["🟡 Unified + Prefetch\n~18–24 GB/s\nNear pinned speed\nSingle pointer simplicity"]
        PAG["🟣 Pageable + cudaMemcpy\n~14–18 GB/s\nExtra staging copy overhead"]
        ZC["🔵 Zero-Copy\n~8–16 GB/s effective\nPer-access PCIe latency\nNo VRAM required"]
    end

    style G   fill:#c0392b,color:#fff,stroke:#922b21
    style PIN fill:#d35400,color:#fff,stroke:#a04000
    style UMP fill:#c8a000,color:#1a1a1a,stroke:#b7950b
    style PAG fill:#7d3c98,color:#fff,stroke:#6c3483
    style ZC  fill:#1f618d,color:#fff,stroke:#154360
```

## 8.7 When to Use Each

```mermaid
flowchart TD
    START["New CUDA kernel — how to manage memory?"]

    Q1{"Data larger\nthan GPU VRAM?"}
    Q2{"Need maximum\ntransfer speed?"}
    Q3{"Access pattern\nknown ahead of time?"}
    Q4{"Prototyping or\nonce-per-run?"}

    ZC["🔵 Zero-Copy\ncudaHostAlloc(mapped)\nGPU reads over PCIe\nNo VRAM consumed"]
    PIN["🟠 Pinned + cudaMemcpyAsync\ncudaMallocHost\nFastest transfers\nAsync stream overlap ✓"]
    UMP["🟡 Unified + Prefetch\ncudaMallocManaged\n+ cudaMemPrefetchAsync\nSimpler code, near-peak speed"]
    UMN["🟣 Unified, no prefetch\ncudaMallocManaged\nEasiest code\nPage faults OK for prototyping"]
    PAG["⬜ Regular malloc + cudaMemcpy\nSimplest possible\nSlightly slower transfers\nFine for small data"]

    START --> Q1
    Q1 -->|"Yes"| ZC
    Q1 -->|"No"| Q2
    Q2 -->|"Yes — production code"| PIN
    Q2 -->|"No"| Q3
    Q3 -->|"Yes — batch workload"| UMP
    Q3 -->|"No"| Q4
    Q4 -->|"Yes — prototyping"| UMN
    Q4 -->|"No"| PAG

    style START fill:#2c3e50,color:#ecf0f1,stroke:#1a252f
    style Q1    fill:#7d3c98,color:#fff,stroke:#6c3483
    style Q2    fill:#7d3c98,color:#fff,stroke:#6c3483
    style Q3    fill:#7d3c98,color:#fff,stroke:#6c3483
    style Q4    fill:#7d3c98,color:#fff,stroke:#6c3483
    style ZC    fill:#1f618d,color:#fff,stroke:#154360
    style PIN   fill:#d35400,color:#fff,stroke:#a04000
    style UMP   fill:#c8a000,color:#1a1a1a,stroke:#b7950b
    style UMN   fill:#1e8449,color:#fff,stroke:#196f3d
    style PAG   fill:#566573,color:#fff,stroke:#4d5656
```

| Approach | Use When |
|----------|---------|
| Regular malloc + cudaMemcpy | Maximum performance, large repeated transfers |
| Pinned (cudaMallocHost) | Same as above but need faster transfers or async copies |
| Unified Memory + prefetch | Simpler code, data access pattern known ahead of time |
| Unified Memory (no prefetch) | Prototyping, irregular access patterns |
| Zero-copy | Data > VRAM, accessed once, or strong CPU-GPU sharing |

## 8.8 Exercises

1. In `01_pinned_memory.cu`, add a 4 GB transfer test. At what size does the bandwidth difference between pageable and pinned stabilize?
2. In `02_unified_memory.cu`, comment out the prefetch calls. Run and observe the slowdown from page faults. How much slower is it?
3. Try running a kernel on unified memory **without** `cudaDeviceSynchronize` before CPU access. What happens? Why?
4. Implement a benchmark using `cudaMemAdvise(data, bytes, cudaMemAdviseSetReadMostly, device)` for a read-only data array. Does it improve performance?

## 8.9 Key Takeaways

- **Pinned memory** eliminates the staging copy for H2D/D2H transfers, giving 15–30% faster transfers.
- **Pinned memory is required** for `cudaMemcpyAsync` to be truly asynchronous.
- **Unified memory** uses a single pointer for CPU+GPU but incurs page fault costs on first access.
- Use **cudaMemPrefetchAsync** to eliminate page faults by migrating data proactively — matches cudaMemcpy speed.
- **Zero-copy** maps CPU memory into GPU space — useful for large data or infrequent access, but bandwidth is PCIe-limited (~16 GB/s vs 1 TB/s VRAM).
- Don't over-use pinned memory — it reduces available physical RAM for the OS.

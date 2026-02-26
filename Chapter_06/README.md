# Chapter 06: CUDA Streams and Concurrency

## 6.1 What is a CUDA Stream?

A **CUDA stream** is an ordered queue of GPU operations (kernel launches, memory copies, events). Operations within a single stream execute in order. Operations in **different streams** may execute concurrently.

The **default stream** (stream 0, or `cudaStreamLegacy`) is special: it synchronizes with all other streams by default. If you mix stream 0 with non-default streams, stream 0 acts as a global barrier.

```
Stream 0 (default):  [Op A]──[Op B]──[Op C]           Sequential
Stream 1:            [Copy H2D]──[Kernel]──[Copy D2H]  ↕ May overlap
Stream 2:            [Copy H2D]──[Kernel]──[Copy D2H]  ↕ with Stream 1
```

## 6.2 Why Use Streams?

Modern GPUs have separate engines for:
- **Compute** (kernel execution)
- **DMA Copy Engine** for H2D transfers
- **DMA Copy Engine** for D2H transfers

These can run **simultaneously**. Without streams, you serialize everything:

```
Sequential (no streams):
[H2D copy] → [Kernel] → [D2H copy] → [H2D copy] → [Kernel] → [D2H copy]

Pipelined (with streams):
Stream 0: [H2D A] → [Kernel A] ────────────── → [D2H A]
Stream 1:         → [H2D B]   → [Kernel B] → [D2H B]
```

The pipelined version overlaps H2D B with Kernel A, and Kernel B with D2H A — significantly improving throughput for batch-style workloads.

## 6.3 Creating and Using Streams

```c
// Create streams
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Launch async operations on a stream
cudaMemcpyAsync(d_dst, h_src, bytes, cudaMemcpyHostToDevice, stream1);
myKernel<<<grid, block, 0, stream1>>>(d_data);          // 3rd param: shared mem
cudaMemcpyAsync(h_dst, d_src, bytes, cudaMemcpyDeviceToHost, stream1);

// Synchronize a single stream
cudaStreamSynchronize(stream1);  // Wait only for stream1 to finish

// Or synchronize all streams
cudaDeviceSynchronize();

// Destroy when done
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

**Critical**: `cudaMemcpyAsync` requires **pinned (page-locked) host memory**. Regular `malloc` memory silently falls back to synchronous behavior.

```c
// Must use pinned memory for true async transfers
float *h_data;
cudaMallocHost(&h_data, bytes);   // Pinned host memory
// ...
cudaFreeHost(h_data);
```

## 6.4 Stream Synchronization Primitives

### cudaStreamSynchronize
Blocks the host until all operations in a specific stream are complete:
```c
cudaStreamSynchronize(stream);
```

### cudaDeviceSynchronize
Blocks the host until **all** GPU operations across all streams are complete:
```c
cudaDeviceSynchronize();
```

### CUDA Events Across Streams
Events can create dependency relationships between streams without blocking the host:
```c
cudaEvent_t event;
cudaEventCreate(&event);

// Record event in stream1 when it reaches this point
cudaEventRecord(event, stream1);

// Make stream2 wait for stream1's event (GPU-side wait, CPU keeps running)
cudaStreamWaitEvent(stream2, event, 0);

cudaEventDestroy(event);
```

### Non-blocking Streams
By default, streams synchronize with the default stream. Create a truly independent stream:
```c
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
```

## 6.5 The Double-Buffering Pipeline Pattern

For processing large datasets in chunks, the double-buffer pattern gives maximum overlap:

```
Stream A: [H2D chunk 0] [Kernel chunk 0]             [D2H chunk 0]
Stream B:              [H2D chunk 1]  [Kernel chunk 1]             [D2H chunk 1]
Stream A:                                             [H2D chunk 2] ...
```

Each chunk alternates between two streams (A and B). While chunk N is being processed by the kernel on stream A, chunk N+1's data is being copied to the GPU on stream B.

This **hides transfer latency** behind computation.

## 6.6 Concurrent Kernel Execution

On GPUs with `concurrentKernels = 1` (almost all modern GPUs), multiple small kernels in different streams can run simultaneously if the GPU has idle SMs:

```c
// These two kernels may execute simultaneously on different SMs
kernelA<<<small_grid, block, 0, stream1>>>(data_a);
kernelB<<<small_grid, block, 0, stream2>>>(data_b);
```

This is useful when a single kernel doesn't saturate all SMs.

## 6.7 Exercises

1. Run `01_streams_basic.cu`. Measure the speedup from using multiple streams. Does it match your expectation based on the H2D/compute/D2H breakdown?
2. In `02_async_overlap.cu`, vary the chunk size (try 1/2, 1/4, 1/8 of total data). How does chunk size affect the overlap efficiency?
3. What happens if you forget `cudaMallocHost` and use regular `malloc` for async transfers? Add a test to verify.
4. Add a third stream to the double-buffering example. Does performance improve? Why or why not?
5. Use `cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)` and observe how it differs from default-flag streams when mixing with stream 0.

## 6.8 Key Takeaways

- Streams are ordered queues; different streams can overlap.
- `cudaMemcpyAsync` **requires pinned host memory** (`cudaMallocHost`).
- The kernel launch 3rd parameter is dynamic shared memory, not stream — stream is the **4th**.
- Use `cudaStreamWaitEvent` for GPU-side stream dependencies without blocking the CPU.
- The double-buffer pipeline pattern is the standard for maximizing throughput on streaming data.
- Don't use the default stream with non-default streams unless you want implicit synchronization.

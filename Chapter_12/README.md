# Chapter 12: Multi-GPU Programming and Advanced Optimization

## 12.1 Multi-GPU Basics

Modern workstations and servers have multiple GPUs. CUDA supports programming across all of them from a single host process.

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

### Data Parallel Multi-GPU

The simplest strategy: split the data evenly and run the same kernel on each GPU:

```
Data: [0 ... N/2-1]   [N/2 ... N-1]
       GPU 0               GPU 1
       kernel              kernel    (concurrent)
Result: merge on CPU
```

Time ≈ single-GPU time / N_gpus (ideal — actual depends on transfer overhead).

## 12.2 Peer-to-Peer (P2P) Access

Without P2P, GPU-to-GPU copies go through CPU RAM:
```
GPU 0 VRAM → PCIe → CPU RAM → PCIe → GPU 1 VRAM   (slow)
```

With P2P enabled (requires NVLink or same PCIe switch):
```
GPU 0 VRAM → NVLink → GPU 1 VRAM   (fast)
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

Your system has RTX 4090 + GTX 1050. These are on different PCIe slots without NVLink — P2P via PCIe may or may not be supported. Check with `01_multi_gpu.cu`.

## 12.3 Advanced Optimization Techniques

### Vectorized Memory Loads (float4)

Instead of loading one float per instruction, load 4 floats at once (128-bit transaction):

```c
// Scalar: 1 load instruction per thread = 4 bytes
float val = data[i];

// Vectorized: 1 load instruction per thread = 16 bytes (4x throughput potential)
float4 val4 = reinterpret_cast<float4*>(data)[i];
float a = val4.x, b = val4.y, c = val4.z, d = val4.w;
```

Vectorized loads:
- Reduce instruction count by 4x
- Ensure 128-bit aligned memory transactions (ideal for coalescing)
- Can significantly improve bandwidth utilization

**Requirement**: array must be 16-byte aligned and size divisible by 4.

### `__ldg()` — Read-only Cache

```c
// Standard global load (goes through L1/L2)
float val = data[i];

// Read-only cache load (texture cache path — better for non-coalesced read-only data)
float val = __ldg(&data[i]);

// Or use __restrict__ pointer — compiler may auto-use __ldg
__global__ void kernel(const float* __restrict__ data, ...) { ... }
```

### Instruction-Level Parallelism (ILP)

Have each thread compute multiple independent values to hide instruction latency:

```c
// ILP=1: sequential dependency chain, hard to hide latency
for (int i = tid; i < n; i += stride)
    sum += data[i];

// ILP=4: 4 independent accumulators — compiler can pipeline
float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
for (int i = tid; i < n; i += stride * 4) {
    s0 += data[i + 0*stride];
    s1 += data[i + 1*stride];
    s2 += data[i + 2*stride];
    s3 += data[i + 3*stride];
}
float sum = s0 + s1 + s2 + s3;
```

### Loop Unrolling

```c
// Manual unroll
#pragma unroll 4
for (int k = 0; k < TILE; k++)
    acc += As[ty][k] * Bs[k][tx];

// Full unroll (TILE must be compile-time constant)
#pragma unroll
for (int k = 0; k < TILE; k++)
    acc += As[ty][k] * Bs[k][tx];
```

## 12.4 Tensor Cores (WMMA API)

Tensor Cores are specialized hardware units for matrix multiply-accumulate (MMA) operations. They operate on 4×4 or 16×16 matrix tiles in a single instruction.

RTX 4090 (Ada Lovelace) Tensor Core throughput:
- FP16/BF16: ~330 TFLOPS
- TF32: ~165 TFLOPS (FP32 input, FP19 accumulation)
- FP64: ~1.8 TFLOPS
- INT8: ~661 TOPS

Using the WMMA API directly:

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

Note: WMMA is a **warp-level** API. All 32 threads in the warp must call these functions together.

## 12.5 Checking PTX Register Usage

```bash
# Compile with register usage report
nvcc -arch=sm_89 -O2 --ptxas-options=-v -o kernel kernel.cu
# Output: ptxas info: Function 'myKernel': 32 registers, 512 bytes smem, ...

# Inspect PTX assembly
nvcc -arch=sm_89 -O2 --ptx -o kernel.ptx kernel.cu

# Inspect SASS (actual GPU assembly)
cuobjdump --dump-sass kernel
```

## 12.6 Exercises

1. Run `01_multi_gpu.cu` and measure the speedup of dual-GPU vs single-GPU vector add. Is it close to 2x?
2. In `02_vectorized_loads.cu`, verify the float4 and scalar results match. What happens if the array size is not a multiple of 4?
3. Run `03_tensor_cores.cu` and compare FP16 WMMA GFLOPS to FP32 CUDA core GFLOPS. How close to the 4x theoretical speedup do you get?
4. Profile `03_tensor_cores.cu` with `ncu`. Check the "SM Throughput" and "Tensor Active" metrics.
5. Implement a batched WMMA GEMM that processes 16 independent 16×16×16 matrix multiplications in a single kernel.

## 12.7 Key Takeaways

- `cudaSetDevice(n)` selects the active GPU for the current thread.
- Multi-GPU: split data, launch kernels on each GPU, merge results — near-linear scaling for compute-bound work.
- `float4` vectorized loads reduce instruction count and improve bandwidth utilization.
- `#pragma unroll` and ILP=4 help hide instruction and memory latency.
- WMMA provides warp-level Tensor Core access for FP16/BF16 matrix multiply.
- For production: use cuBLAS (already uses Tensor Cores) rather than raw WMMA.

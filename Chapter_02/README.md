# Chapter 02: The CUDA Programming Model — Kernels, Threads, and Indexing

## 2.1 Writing a Kernel

A **kernel** is a C function decorated with `__global__` that runs on the GPU. When launched, CUDA creates many copies of this function running simultaneously — one per thread.

```c
// A simple kernel that squares each element of an array
__global__ void squareKernel(float *d_out, float *d_in, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
    if (i < n)                                        // bounds check!
        d_out[i] = d_in[i] * d_in[i];
}
```

### Function Qualifiers

CUDA adds three qualifiers that control where a function is compiled and who can call it:

```mermaid
graph LR
    subgraph HOST["🖥️  HOST — runs on CPU"]
        H["__host__ fn()\nRegular C/C++\n(implicit default)"]
    end
    subgraph DEVICE["🎮  DEVICE — runs on GPU"]
        K["__global__ kernel()\nEntry point from CPU\nmust return void"]
        D["__device__ fn()\nHelper called from kernels\nGPU-only"]
    end
    HD["__host__ __device__ fn()\nCompiled for BOTH\nMath helpers, utilities"]

    H -->|"launches via\n&lt;&lt;&lt;grid, block&gt;&gt;&gt;"| K
    H -->|"calls normally"| HD
    K -->|"calls"| D
    K -->|"calls"| HD
    D -->|"calls"| HD

    style HOST fill:#0d2137,color:#aed6f1,stroke:#2980b9
    style DEVICE fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style H fill:#2471a3,color:#fff,stroke:#1a5276
    style K fill:#e74c3c,color:#fff,stroke:#c0392b
    style D fill:#1e8449,color:#fff,stroke:#196f3d
    style HD fill:#7d3c98,color:#fff,stroke:#6c3483
```

| Qualifier | Callable from | Runs on | Notes |
|-----------|--------------|---------|-------|
| `__global__` | Host (CPU) | Device (GPU) | Must return `void`, kernel entry point |
| `__device__` | Device only | Device only | Helper functions called from kernels |
| `__host__` | Host only | Host only | Normal C/C++ (default, rarely needed explicitly) |
| `__host__ __device__` | Both | Both | Compiled for both — useful for math helpers |

## 2.2 Thread Indexing — 1D

The most common pattern maps one thread to one element of an array.

For a 1D kernel launch with `N` elements, the **global thread index** for thread `t` in block `b` with block size `B` is:

```
i = b * B + t
  = blockIdx.x * blockDim.x + threadIdx.x
```

### Why Bounds Checking Matters

The number of threads launched is always rounded up to a full block. The last block often has threads that map beyond the array end:

```diff
  kernel<<<3, 4>>>()  →  12 threads total,  but N = 10 elements

+ Thread  0  (i= 0):  d_out[0]  = d_in[0]  * d_in[0]   ✓ in bounds
+ Thread  1  (i= 1):  d_out[1]  = d_in[1]  * d_in[1]   ✓ in bounds
+ Thread  2  (i= 2):  d_out[2]  = d_in[2]  * d_in[2]   ✓ in bounds
+ Thread  3  (i= 3):  d_out[3]  = d_in[3]  * d_in[3]   ✓ in bounds
+ Thread  4  (i= 4):  d_out[4]  = d_in[4]  * d_in[4]   ✓ in bounds
+ Thread  5  (i= 5):  d_out[5]  = d_in[5]  * d_in[5]   ✓ in bounds
+ Thread  6  (i= 6):  d_out[6]  = d_in[6]  * d_in[6]   ✓ in bounds
+ Thread  7  (i= 7):  d_out[7]  = d_in[7]  * d_in[7]   ✓ in bounds
+ Thread  8  (i= 8):  d_out[8]  = d_in[8]  * d_in[8]   ✓ in bounds
+ Thread  9  (i= 9):  d_out[9]  = d_in[9]  * d_in[9]   ✓ in bounds
- Thread 10  (i=10):  d_out[10] = ???  OUT OF BOUNDS — memory corruption! 🔥
- Thread 11  (i=11):  d_out[11] = ???  OUT OF BOUNDS — memory corruption! 🔥

  Guard with:  if (i < n)  ← skips threads 10 and 11 safely ✓
```

## 2.3 Thread Indexing — 2D

For 2D data (images, matrices), use 2D grids and blocks. Each thread computes `(col, row)` coordinates:

```c
// 2D kernel — process a W x H image
__global__ void process2D(float *data, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // x-coordinate
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // y-coordinate

    if (col < width && row < height) {
        int idx = row * width + col;  // row-major linearization
        data[idx] *= 2.0f;
    }
}
```

Launch with a 2D block and grid:
```c
dim3 block(16, 16);           // 16x16 = 256 threads per block
dim3 grid((W + 15) / 16,      // ceil(W/16) blocks in x
          (H + 15) / 16);     // ceil(H/16) blocks in y
process2D<<<grid, block>>>(d_data, W, H);
```

### 2D Grid / Block Layout

```mermaid
graph TD
    LAUNCH["process2D &lt;&lt;&lt; grid={3,2}, block={4,4} &gt;&gt;&gt;()"]

    subgraph GRID["GRID — 3×2 = 6 blocks  (covers full W×H image)"]
        subgraph GR0["blockIdx.y = 0  (top row of blocks)"]
            B00["Block (0,0)\ncols 0–3\nrows 0–3"]
            B10["Block (1,0)\ncols 4–7\nrows 0–3"]
            B20["Block (2,0)\ncols 8–11\nrows 0–3"]
        end
        subgraph GR1["blockIdx.y = 1  (bottom row of blocks)"]
            B01["Block (0,1)\ncols 0–3\nrows 4–7"]
            B11["Block (1,1)\ncols 4–7\nrows 4–7"]
            B21["Block (2,1)\ncols 8–11\nrows 4–7"]
        end
    end

    subgraph BLOCK["Block (0,0) zoomed — 4×4 = 16 threads"]
        subgraph TRY0["threadIdx.y=0"]
            T00["(0,0)\ncol=0,row=0"] --- T10["(1,0)\ncol=1,row=0"] --- T20["(2,0)\ncol=2,row=0"] --- T30["(3,0)\ncol=3,row=0"]
        end
        subgraph TRY1["threadIdx.y=1"]
            T01["(0,1)\ncol=0,row=1"] --- T11["(1,1)\ncol=1,row=1"] --- T21["(2,1)\ncol=2,row=1"] --- T31["(3,1)\ncol=3,row=1"]
        end
    end

    LAUNCH --> GRID
    B00 -->|"zoom in"| BLOCK

    style LAUNCH fill:#2c3e50,color:#ecf0f1,stroke:#1a252f
    style GRID fill:#0d1b2a,color:#aed6f1,stroke:#2980b9
    style GR0 fill:#0d2137,color:#aed6f1,stroke:#2471a3
    style GR1 fill:#0d2137,color:#aed6f1,stroke:#2471a3
    style BLOCK fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style TRY0 fill:#0d2a0d,color:#a9dfbf,stroke:#1e8449
    style TRY1 fill:#0d2a0d,color:#a9dfbf,stroke:#1e8449
    style B00 fill:#e74c3c,color:#fff,stroke:#c0392b
    style B10 fill:#2471a3,color:#fff,stroke:#1a5276
    style B20 fill:#2471a3,color:#fff,stroke:#1a5276
    style B01 fill:#2471a3,color:#fff,stroke:#1a5276
    style B11 fill:#2471a3,color:#fff,stroke:#1a5276
    style B21 fill:#2471a3,color:#fff,stroke:#1a5276
    style T00 fill:#1e8449,color:#fff,stroke:#196f3d
    style T10 fill:#1e8449,color:#fff,stroke:#196f3d
    style T20 fill:#1e8449,color:#fff,stroke:#196f3d
    style T30 fill:#1e8449,color:#fff,stroke:#196f3d
    style T01 fill:#1e8449,color:#fff,stroke:#196f3d
    style T11 fill:#1e8449,color:#fff,stroke:#196f3d
    style T21 fill:#1e8449,color:#fff,stroke:#196f3d
    style T31 fill:#1e8449,color:#fff,stroke:#196f3d
```

The `dim3` type is a struct with `.x`, `.y`, `.z` fields (default `.z = 1`).

## 2.4 Choosing Block Size

Block size is one of the most important tuning parameters.

```mermaid
flowchart TD
    WARP["⚠️  Warp size = 32\n(hardware minimum — never change this)"]

    WARP -->|"Block size must be\na multiple of"| MULT["Multiples of 32"]
    MULT --> S128["128 threads\n4 warps\nGood when kernels\nuse many registers"]
    MULT --> S256["256 threads  ✅ DEFAULT\n8 warps\nSafe, well-balanced\nstart here"]
    MULT --> S512["512 threads\n16 warps\nHigh occupancy\nwatch shared memory"]
    MULT --> S1024["1024 threads\n32 warps  (MAX)\nMaximum parallelism\nleast SM flexibility"]

    WARP -->|"Block size must be ≤"| MAX["1024 threads\n(hardware limit)"]

    style WARP fill:#c0392b,color:#fff,stroke:#922b21
    style MULT fill:#1f618d,color:#fff,stroke:#154360
    style MAX fill:#7d3c98,color:#fff,stroke:#6c3483
    style S128 fill:#2c3e50,color:#aed6f1,stroke:#2980b9
    style S256 fill:#1e8449,color:#fff,stroke:#196f3d
    style S512 fill:#2c3e50,color:#aed6f1,stroke:#2980b9
    style S1024 fill:#2c3e50,color:#aed6f1,stroke:#2980b9
```

Rules:
1. Block size must be a **multiple of 32** (the warp size). Non-multiples waste hardware.
2. Common choices: **128, 256, 512** (256 is a safe default).
3. Max threads per block is typically **1024**.
4. Larger blocks share more shared memory but limit the number of blocks per SM.

For 2D blocks: 16×16 = 256 and 32×32 = 1024 are both common.

## 2.5 The Vector Addition Example

Vector addition is the "Hello, World!" of GPU computing:

```mermaid
graph LR
    subgraph HOST["🖥️  HOST"]
        HA["h_A[]\n[1.0, 1.0, 1.0, ...]"]
        HB["h_B[]\n[2.0, 2.0, 2.0, ...]"]
        HC["h_C[]\n[3.0, 3.0, 3.0, ...]"]
    end

    subgraph DEVICE["🎮  DEVICE"]
        DA["d_A[]"]
        DB["d_B[]"]
        DC["d_C[]"]
        subgraph KERNEL["vecAdd kernel  (N threads in parallel)"]
            K0["Thread 0\nC[0]=A[0]+B[0]"]
            K1["Thread 1\nC[1]=A[1]+B[1]"]
            K2["Thread 2\nC[2]=A[2]+B[2]"]
            KN["Thread N-1\nC[N-1]=A[N-1]+B[N-1]"]
        end
    end

    HA -->|"H→D memcpy"| DA
    HB -->|"H→D memcpy"| DB
    DA --> KERNEL
    DB --> KERNEL
    KERNEL --> DC
    DC -->|"D→H memcpy"| HC

    style HOST fill:#0d2137,color:#aed6f1,stroke:#2980b9
    style DEVICE fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style KERNEL fill:#1e3a0d,color:#a9dfbf,stroke:#1e8449
    style HA fill:#2471a3,color:#fff,stroke:#1a5276
    style HB fill:#2471a3,color:#fff,stroke:#1a5276
    style HC fill:#2471a3,color:#fff,stroke:#1a5276
    style DA fill:#1e8449,color:#fff,stroke:#196f3d
    style DB fill:#1e8449,color:#fff,stroke:#196f3d
    style DC fill:#1e8449,color:#fff,stroke:#196f3d
    style K0 fill:#145a32,color:#a9dfbf,stroke:#1e8449
    style K1 fill:#145a32,color:#a9dfbf,stroke:#1e8449
    style K2 fill:#145a32,color:#a9dfbf,stroke:#1e8449
    style KN fill:#145a32,color:#a9dfbf,stroke:#1e8449
```

Properties that make it ideal for learning:
- **Perfectly parallel** — each element `C[i]` is fully independent
- **Memory-bound** — most time is spent on memory, not computation (arithmetic intensity ≈ 0.17 FLOP/byte)
- Simple enough to focus on the CUDA mechanics

See `01_vector_add.cu` for the full, commented example with timing.

## 2.6 Thread ID Patterns — Summary

```c
// 1D grid, 1D blocks
int i = blockIdx.x * blockDim.x + threadIdx.x;

// 2D grid, 2D blocks
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
int i   = row * width + col;

// 3D grid, 3D blocks (for volumetric data)
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
int i = z * (width * height) + y * width + x;

// 1D grid but operating on 2D data via striding
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for (int i = tid; i < n; i += stride) { ... }  // grid-stride loop
```

### The Grid-Stride Loop

The grid-stride loop is a robust pattern when the grid may be smaller than the data size. Each thread processes multiple elements by stepping by the total grid width:

```mermaid
graph LR
    subgraph DATA["Array with N=24 elements"]
        E0["[0]"] --- E1["[1]"] --- E2["[2]"] --- E3["[3]"] --- E4["[4]"] --- E5["[5]"]
        E6["[6]"] --- E7["[7]"] --- E8["[8]"] --- E9["[9]"] --- E10["[10]"] --- E11["[11]"]
        E12["[12]"] --- E13["[13]"] --- E14["[14]"] --- E15["[15]"] --- E16["[16]"] --- E17["[17]"]
        E18["[18]"] --- E19["[19]"] --- E20["[20]"] --- E21["[21]"] --- E22["[22]"] --- E23["[23]"]
    end

    subgraph GRID["Grid: 2 blocks × 4 threads = 8 threads  (stride = 8)"]
        T0["Thread 0\ni=0,8,16"]
        T1["Thread 1\ni=1,9,17"]
        T2["Thread 2\ni=2,10,18"]
        T3["Thread 3\ni=3,11,19"]
        T4["Thread 4\ni=4,12,20"]
        T5["Thread 5\ni=5,13,21"]
        T6["Thread 6\ni=6,14,22"]
        T7["Thread 7\ni=7,15,23"]
    end

    T0 -->|"pass 1"| E0
    T0 -->|"pass 2"| E8
    T0 -->|"pass 3"| E16
    T4 -->|"pass 1"| E4
    T4 -->|"pass 2"| E12
    T4 -->|"pass 3"| E20

    style DATA fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style GRID fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style T0 fill:#e74c3c,color:#fff,stroke:#c0392b
    style T1 fill:#2471a3,color:#fff,stroke:#1a5276
    style T2 fill:#2471a3,color:#fff,stroke:#1a5276
    style T3 fill:#2471a3,color:#fff,stroke:#1a5276
    style T4 fill:#e67e22,color:#fff,stroke:#ca6f1e
    style T5 fill:#2471a3,color:#fff,stroke:#1a5276
    style T6 fill:#2471a3,color:#fff,stroke:#1a5276
    style T7 fill:#2471a3,color:#fff,stroke:#1a5276
```

The grid-stride loop is also more friendly to compiler optimizations and works correctly regardless of how large `N` is relative to the grid.

## 2.7 Memory Management

```mermaid
sequenceDiagram
    participant CPU as 🖥️ HOST (CPU)
    participant GPU as 🎮 DEVICE (GPU / VRAM)

    Note over CPU,GPU: Step 1 — Allocate device memory
    CPU->>GPU: cudaMalloc(&d_ptr, n * sizeof(float))
    activate GPU
    GPU-->>CPU: d_ptr (device address)
    deactivate GPU

    Note over CPU,GPU: Step 2 — Copy input data to GPU
    CPU->>GPU: cudaMemcpy(d_ptr, h_ptr, bytes, HostToDevice)
    activate GPU
    GPU-->>CPU: done
    deactivate GPU

    Note over CPU,GPU: Step 3 — Launch kernel
    CPU-)GPU: kernel<<<grid, block>>>(d_ptr, n)
    activate GPU
    Note over GPU: Runs asynchronously
    CPU->>GPU: cudaDeviceSynchronize()
    GPU-->>CPU: complete ✓
    deactivate GPU

    Note over CPU,GPU: Step 4 — Copy results back
    CPU->>GPU: cudaMemcpy(h_ptr, d_ptr, bytes, DeviceToHost)
    activate GPU
    GPU-->>CPU: results returned
    deactivate GPU

    Note over CPU,GPU: Step 5 — Free device memory
    CPU->>GPU: cudaFree(d_ptr)
```

```c
float *d_ptr;                           // device pointer (prefix d_ by convention)
cudaMalloc(&d_ptr, n * sizeof(float));  // allocate on GPU

// cudaMemcpy(dst, src, bytes, direction)
cudaMemcpy(d_ptr, h_ptr, n * sizeof(float), cudaMemcpyHostToDevice);   // CPU → GPU
cudaMemcpy(h_ptr, d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);   // GPU → CPU

cudaFree(d_ptr);                        // free GPU memory
```

## 2.8 Exercises

1. Modify `01_vector_add.cu` to compute `C[i] = A[i] * A[i] + B[i] * B[i]` (element-wise squared norm).
2. In `02_thread_indexing.cu`, change the block size from 256 to 128 and 512. Does the output change? Does timing change?
3. Write a kernel that computes the element-wise product of two vectors (Hadamard product).
4. In `02_thread_indexing.cu`, modify it to transpose a matrix: `out[col][row] = in[row][col]`. Run it — is it correct? (Hint: transposition is tricky, we'll optimize it in Chapter 03.)
5. What happens if you launch a kernel with 0 blocks? What about 0 threads per block? Try it.

## 2.9 Key Takeaways

- Kernels are launched with `<<<numBlocks, threadsPerBlock>>>` syntax.
- Use `blockIdx`, `blockDim`, and `threadIdx` to compute each thread's unique global index.
- **Always bounds-check**: the total thread count may exceed your data size.
- Block size should be a **multiple of 32**; 256 is a safe default.
- The **grid-stride loop** pattern handles arbitrary data sizes robustly.
- `cudaMalloc` / `cudaMemcpy` / `cudaFree` are the three fundamental memory operations.

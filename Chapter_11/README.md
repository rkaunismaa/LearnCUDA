# Chapter 11: CUDA with Python — PyTorch and CuPy

## 11.1 Python CUDA Ecosystem

Python is the dominant language for GPU computing in ML and scientific computing. Several libraries expose GPU power at different abstraction levels:

```mermaid
graph TD
    subgraph PYTHON["Python CUDA Ecosystem — Abstraction Levels"]
        PT["PyTorch\\nDeep learning + tensor ops\\nAutograd, AMP, Profiler\\nC++/CUDA backend, Python API\\n→ best for ML workloads"]
        CP["CuPy\\nNumPy-compatible GPU arrays\\nRawKernel for custom CUDA C\\n→ best for scientific computing"]
        NB["Numba\\nJIT-compile Python → CUDA\\n@cuda.jit decorator\\n→ best for porting Python loops"]
        PC["PyCUDA\\nCtypes-level CUDA wrapping\\nFine-grained memory control\\n→ best for low-level control"]
    end

    subgraph UNDER["Under the Hood"]
        CB2["cuBLAS / cuFFT\\ncuDNN / cuSPARSE"]
        DRV["CUDA Driver / Runtime\\nlibcuda.so, libcudart.so"]
        HW["🎮 GPU Hardware"]
    end

    PT  --> CB2
    CP  --> CB2
    NB  --> DRV
    PC  --> DRV
    CB2 --> DRV
    DRV --> HW

    style PYTHON fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style UNDER  fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style PT     fill:#d35400,color:#fff,stroke:#a04000
    style CP     fill:#1f618d,color:#fff,stroke:#154360
    style NB     fill:#7d3c98,color:#fff,stroke:#6c3483
    style PC     fill:#c0392b,color:#fff,stroke:#922b21
    style CB2    fill:#1e8449,color:#fff,stroke:#196f3d
    style DRV    fill:#1e8449,color:#fff,stroke:#196f3d
    style HW     fill:#2c3e50,color:#ecf0f1,stroke:#1a252f
```

This chapter focuses on **PyTorch** and **CuPy** — the two most widely used libraries.

## 11.2 Setup

```bash
# Activate the virtual environment
source /home/rob/PythonEnvironments/LearnCUDA/.learncuda/bin/activate

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install CuPy for CUDA 12.x
pip install cupy-cuda12x

# Install helpers
pip install numpy matplotlib jupyter
```

## 11.3 PyTorch CUDA Basics

### Device Management

```python
import torch

torch.cuda.is_available()        # True if CUDA GPU present
torch.cuda.device_count()        # Number of GPUs
torch.cuda.get_device_name(0)    # "NVIDIA GeForce RTX 4090"

# Create tensor on GPU
x = torch.zeros(1000, device='cuda')
x_cpu = torch.randn(1000)
x_gpu = x_cpu.to('cuda')
x_back = x_gpu.cpu()
```

```mermaid
graph LR
    subgraph HOST["🖥️ Host (CPU / RAM)"]
        HC["torch.randn(1000)\\nx_cpu: float32 tensor"]
    end
    subgraph GPU0["🎮 GPU 0 — RTX 4090 (VRAM)"]
        GC["x_gpu = x_cpu.to('cuda')\\ndevice='cuda:0'"]
    end
    subgraph GPU1["🎮 GPU 1 — GTX 1050 (VRAM)"]
        G1["with torch.cuda.device(1):\\n    x = torch.randn(100, device='cuda')"]
    end

    HC -->|"H2D copy\\n.to('cuda')"| GC
    GC -->|"D2H copy\\n.cpu()"| HC
    HC -->|"H2D copy\\ndevice='cuda:1'"| G1

    style HOST fill:#1f618d,color:#fff,stroke:#154360
    style GPU0 fill:#1e8449,color:#fff,stroke:#196f3d
    style GPU1 fill:#7d3c98,color:#fff,stroke:#6c3483
    style HC   fill:#154360,color:#fff,stroke:#1f618d
    style GC   fill:#196f3d,color:#fff,stroke:#1e8449
    style G1   fill:#6c3483,color:#fff,stroke:#7d3c98
```

### Accurate Timing (GPU ops are async!)

```diff
  WRONG: GPU ops are async — CPU timer measures scheduling overhead only

- import time
- t0 = time.time()
- C = torch.matmul(A, B)
- t1 = time.time()             # Returns BEFORE the GPU finishes! ✗
- elapsed = (t1 - t0) * 1000   # Measures ~0.05 ms — the kernel launch, not the work

  CORRECT: CUDA events synchronize to the actual GPU completion

+ start = torch.cuda.Event(enable_timing=True)
+ end   = torch.cuda.Event(enable_timing=True)
+ start.record()
+ C = torch.matmul(A, B)
+ end.record()
+ torch.cuda.synchronize()        # Block CPU until GPU finishes ✓
+ ms = start.elapsed_time(end)    # True GPU time in milliseconds ✓
```

### Memory Tracking

```python
torch.cuda.memory_allocated()       # Current bytes used
torch.cuda.max_memory_allocated()   # Peak bytes since last reset
torch.cuda.reset_peak_memory_stats()

# Context manager for specific GPU
with torch.cuda.device(1):
    x = torch.randn(100, device='cuda')  # Goes to GPU 1
```

## 11.4 Mixed Precision

Tensor Cores accelerate FP16/BF16/TF32 matmul, delivering up to 4× more throughput than FP32 CUDA cores:

```mermaid
graph LR
    subgraph DTYPES["RTX 4090 Matmul Throughput by dtype"]
        FP32["FP32\\n~82 TFLOPS\\nCUDA cores only\\nhigh precision"]
        TF32["TF32\\n~165 TFLOPS\\nTensor Cores\\nsame output as FP32 for most ops"]
        FP16["FP16\\n~330 TFLOPS\\nTensor Cores\\nhalf memory bandwidth"]
        BF16["BF16\\n~330 TFLOPS\\nTensor Cores\\nbetter range than FP16"]
    end

    FP32 -->|"allow_tf32=True\\n(default ≥ 1.7)"| TF32
    TF32 -->|"dtype=torch.float16\\nor autocast"| FP16
    TF32 -->|"dtype=torch.bfloat16\\nor autocast"| BF16

    style DTYPES fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style FP32   fill:#c0392b,color:#fff,stroke:#922b21
    style TF32   fill:#d35400,color:#fff,stroke:#a04000
    style FP16   fill:#1e8449,color:#fff,stroke:#196f3d
    style BF16   fill:#1e8449,color:#fff,stroke:#196f3d
```

```python
# Enable TF32 globally (default in PyTorch ≥ 1.7)
torch.backends.cuda.matmul.allow_tf32 = True

# FP16 operations
A = torch.randn(2048, 2048, device='cuda', dtype=torch.float16)
B = torch.randn(2048, 2048, device='cuda', dtype=torch.float16)
C = torch.matmul(A, B)

# Automatic Mixed Precision (AMP) for training
with torch.autocast(device_type='cuda', dtype=torch.float16):
    loss = model(input)  # Automatically uses FP16 where safe
```

```mermaid
flowchart LR
    subgraph AMP["Automatic Mixed Precision (AMP) — autocast"]
        OP1["Linear / Conv\\nMatrix multiply\\nFP16 ✓ (fast)"]
        OP2["Loss function\\nBatchNorm\\nFP32 ✓ (stable)"]
        OP3["Gradient accumulation\\nOptimizer step\\nFP32 master weights ✓"]
    end

    INPUT["FP32 input"] --> OP1
    OP1 -->|"autocast casts\\ndown automatically"| OP2
    OP2 --> OP3
    OP3 -->|"GradScaler avoids\\nFP16 underflow"| OP1

    style AMP   fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style INPUT fill:#1f618d,color:#fff,stroke:#154360
    style OP1   fill:#1e8449,color:#fff,stroke:#196f3d
    style OP2   fill:#d35400,color:#fff,stroke:#a04000
    style OP3   fill:#1f618d,color:#fff,stroke:#154360
```

## 11.5 CuPy

CuPy is a drop-in NumPy replacement that runs on the GPU. Most NumPy code works unchanged by replacing `np` with `cp`:

```mermaid
graph LR
    subgraph NP["🖥️ NumPy (CPU)"]
        NPA["import numpy as np\\nnp.zeros(1000)\\nnp.random.randn(1000)\\nnp.sqrt(x**2 + y**2)\\nnp.sum(z)"]
    end
    subgraph CPY["🎮 CuPy (GPU)"]
        CPA["import cupy as cp\\ncp.zeros(1000)\\ncp.random.randn(1000)\\ncp.sqrt(x**2 + y**2)\\ncp.sum(z)"]
    end

    NPA -->|"cp.asarray(x_np)\\nCPU → GPU"| CPA
    CPA -->|"cp.asnumpy(x_cp)\\nor x.get()\\nGPU → CPU"| NPA

    style NP  fill:#1f618d,color:#fff,stroke:#154360
    style CPY fill:#1e8449,color:#fff,stroke:#196f3d
    style NPA fill:#154360,color:#fff,stroke:#1f618d
    style CPA fill:#196f3d,color:#fff,stroke:#1e8449
```

```python
import cupy as cp
import numpy as np

# Create arrays
x = cp.zeros(1000)
y = cp.random.randn(1000)

# Works exactly like NumPy
z = cp.sqrt(x**2 + y**2)
s = cp.sum(z)

# Transfer
x_np = cp.asnumpy(x)         # GPU → CPU  (same as x.get())
y_cp = cp.asarray(x_np)      # CPU → GPU
```

### CuPy RawKernel — CUDA C Inside Python

```mermaid
sequenceDiagram
    participant PY  as 🐍 Python
    participant RK  as cp.RawKernel
    participant PTX as CUDA Compiler (nvcc)
    participant GPU as 🎮 GPU

    PY  ->> RK:  cp.RawKernel(cuda_source_string, 'vector_add')
    RK  ->> PTX: compile CUDA C → PTX / cubin (cached)
    PTX -->> RK: compiled kernel object

    PY  ->> GPU: a = cp.ones(1024, dtype=cp.float32)
    PY  ->> GPU: b = cp.ones(1024, dtype=cp.float32)
    PY  ->> GPU: c = cp.zeros(1024, dtype=cp.float32)

    PY  ->> RK:  add_kernel((4,), (256,), (a, b, c, 1024))
    Note over RK: (grid_dims), (block_dims), args
    RK  ->> GPU: launch vector_add<<<(4,),(256,)>>>(a, b, c, 1024)
    GPU -->> PY: result in c (stays on GPU as cupy array)
```

```python
add_kernel = cp.RawKernel(r'''
extern "C" __global__
void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
''', 'vector_add')

a = cp.ones(1024, dtype=cp.float32)
b = cp.ones(1024, dtype=cp.float32)
c = cp.zeros(1024, dtype=cp.float32)
add_kernel((4,), (256,), (a, b, c, 1024))   # (grid,), (block,), args
```

## 11.6 Custom PyTorch CUDA Extensions

For production-quality custom ops, PyTorch's C++ extension API compiles CUDA C and exposes it as a native Python function with autograd support:

```mermaid
graph TD
    subgraph EXT["Custom PyTorch Extension — Build Flow"]
        PY2["Python caller\\nmy_ext.vector_add_scale(a, b, scale)"]
        BIND["pybind11 binding\\nmy_ext.cpp\\nTORCH_EXTENSION"]
        CU["CUDA kernel\\nmy_kernel.cu\\n__global__ fused_op(...)"]
        BUILD["setup.py\\nfrom torch.utils.cpp_extension\\nimport CUDAExtension, BuildExtension\\npip install -e ."]
    end

    PY2  --> BIND
    BIND --> CU
    BUILD -->|"compiles"| BIND
    BUILD -->|"compiles"| CU

    style EXT   fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style PY2   fill:#d35400,color:#fff,stroke:#a04000
    style BIND  fill:#7d3c98,color:#fff,stroke:#6c3483
    style CU    fill:#1e8449,color:#fff,stroke:#196f3d
    style BUILD fill:#1f618d,color:#fff,stroke:#154360
```

See `03_torch_custom_extension/` for a working example:

```bash
cd 03_torch_custom_extension
pip install -e .
python test_extension.py
```

The extension implements a fused `vector_add + scale` operation using CUDA, exposed to Python via PyTorch's C++ extension API.

## 11.7 PyTorch Profiler

```mermaid
flowchart TD
    subgraph PROF["torch.profiler Workflow"]
        REC["with torch.profiler.profile(\\n    activities=[CPU, CUDA],\\n    record_shapes=True,\\n    with_flops=True,\\n) as prof:\\n    result = model(input)"]
        TABLE["prof.key_averages()\\n.table(sort_by='cuda_time_total')\\n→ shows top CUDA ops by time"]
        TRACE["prof.export_chrome_trace('trace.json')\\nOpen in chrome://tracing\\n→ visual timeline of all ops"]
    end

    REC --> TABLE
    REC --> TRACE

    subgraph METRICS["What to look for"]
        M1["cuda_time_total  — total GPU time per op"]
        M2["flops           — measured FLOPS vs theoretical"]
        M3["self_cpu_time   — CPU overhead / launch latency"]
        M4["input_shapes    — verify expected tensor dims"]
    end

    TABLE --> METRICS

    style PROF    fill:#1c2833,color:#85c1e9,stroke:#2e86c1
    style METRICS fill:#0d230d,color:#a9dfbf,stroke:#27ae60
    style REC     fill:#7d3c98,color:#fff,stroke:#6c3483
    style TABLE   fill:#d35400,color:#fff,stroke:#a04000
    style TRACE   fill:#1f618d,color:#fff,stroke:#154360
    style M1      fill:#1e8449,color:#fff,stroke:#196f3d
    style M2      fill:#1e8449,color:#fff,stroke:#196f3d
    style M3      fill:#1e8449,color:#fff,stroke:#196f3d
    style M4      fill:#1e8449,color:#fff,stroke:#196f3d
```

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True,
) as prof:
    result = model(input)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Python vs C CUDA — When to Use Each

```mermaid
flowchart TD
    START["New GPU computation task"]

    START --> Q1{"Does PyTorch / CuPy\\nalready have it?"}
    Q1 -->|Yes| USE_LIB["Use the library function\\ntorch.matmul, cp.fft.fft, etc."]

    Q1 -->|No| Q2{"Is it a simple\\nelement-wise op?"}
    Q2 -->|Yes| CUPY_RAW["CuPy RawKernel\\n— fast to write, no build step"]

    Q2 -->|No| Q3{"Need autograd?\\n(backprop through it)"}
    Q3 -->|Yes| TORCH_EXT["PyTorch C++ Extension\\ntorch.utils.cpp_extension\\n— pybind11 + CUDA C"]

    Q3 -->|No| Q4{"Performance critical?\\nProduction code?"}
    Q4 -->|Yes| CUDA_C["Write CUDA C directly\\n(Chapters 1–9 techniques)"]
    Q4 -->|No| CUPY_RAW2["CuPy RawKernel\\n— good enough for research"]

    style START      fill:#7d3c98,color:#fff,stroke:#6c3483
    style Q1         fill:#1f618d,color:#fff,stroke:#154360
    style Q2         fill:#1f618d,color:#fff,stroke:#154360
    style Q3         fill:#1f618d,color:#fff,stroke:#154360
    style Q4         fill:#1f618d,color:#fff,stroke:#154360
    style USE_LIB    fill:#1e8449,color:#fff,stroke:#196f3d
    style CUPY_RAW   fill:#1e8449,color:#fff,stroke:#196f3d
    style CUPY_RAW2  fill:#1e8449,color:#fff,stroke:#196f3d
    style TORCH_EXT  fill:#d35400,color:#fff,stroke:#a04000
    style CUDA_C     fill:#c0392b,color:#fff,stroke:#922b21
```

## 11.8 Exercises

1. Run `01_torch_cuda_basics.py`. Note the GFLOPS for FP32 matmul. Then change to `torch.float16` — does it reach the Tensor Core peak of ~330 TFLOPS?
2. In `02_cupy_basics.py`, write a `RawKernel` for the 1D stencil from Chapter 03 and benchmark it against `cp.convolve`.
3. Add a backward pass to the custom extension by implementing the gradient formula for `z = scale * (a + b)`.
4. Use `torch.profiler` to profile 10 iterations of matmul and inspect the resulting trace in Chrome's trace viewer (`chrome://tracing`).

## 11.9 Key Takeaways

- PyTorch and CuPy provide efficient high-level GPU access without writing CUDA C directly.
- Always use CUDA events or `torch.cuda.synchronize()` for accurate timing — CPU timers measure scheduling, not GPU execution.
- FP16/BF16 Tensor Cores can deliver **4× more throughput** than FP32 CUDA cores for matmul.
- CuPy `RawKernel` embeds CUDA C in Python — great for quick custom operations without a build system.
- PyTorch custom extensions (`torch.utils.cpp_extension`) for production-quality custom ops with autograd support.
- **Decision order**: library function → CuPy RawKernel → PyTorch extension → raw CUDA C.

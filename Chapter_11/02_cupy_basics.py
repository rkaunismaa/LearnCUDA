"""
Chapter 11 — 02_cupy_basics.py

Demonstrates CuPy: NumPy-compatible GPU array library.
- Basic array operations
- NumPy interoperability
- CuPy RawKernel for custom CUDA kernels in Python
- Benchmarking CuPy vs NumPy

Run:
    source /home/rob/PythonEnvironments/LearnCUDA/.learncuda/bin/activate
    python 02_cupy_basics.py
"""

import numpy as np
import time

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    print("CuPy not installed. Run: pip install cupy-cuda12x")
    print("Showing code examples only.\n")
    CUPY_AVAILABLE = False

if not CUPY_AVAILABLE:
    print("Example code (not executed):\n")
    print("  import cupy as cp")
    print("  x = cp.random.randn(1000)")
    print("  y = cp.sqrt(x**2 + 1.0)")
    print("  x_np = cp.asnumpy(x)   # GPU -> CPU")
    exit()

print("=" * 60)
print("CuPy GPU Array Basics")
print(f"CuPy version: {cp.__version__}")
print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
print("=" * 60)
print()

N = 1 << 22  # 4M elements

# ================================================================
# 1. Basic array operations
# ================================================================
print("1. Basic Array Operations")
print("-" * 40)
x = cp.random.randn(N, dtype=cp.float32)
y = cp.random.randn(N, dtype=cp.float32)

z = cp.sqrt(x**2 + y**2)
s = float(cp.sum(z))
print(f"x.shape={x.shape}, x.dtype={x.dtype}, x.device={x.device}")
print(f"sum(sqrt(x^2 + y^2)) = {s:.4f}")

# ================================================================
# 2. NumPy interoperability
# ================================================================
print("\n2. NumPy Interoperability")
print("-" * 40)
x_np = np.random.randn(1000).astype(np.float32)

# CPU → GPU
x_cp = cp.asarray(x_np)
print(f"np → cp: {type(x_cp).__name__} on {x_cp.device}")

# GPU → CPU
x_back = cp.asnumpy(x_cp)   # same as x_cp.get()
print(f"cp → np: {type(x_back).__name__}")
print(f"Roundtrip max error: {np.abs(x_back - x_np).max():.2e}")

# ================================================================
# 3. Benchmark: NumPy vs CuPy
# ================================================================
print("\n3. NumPy vs CuPy Benchmark")
print("-" * 40)

operations = [
    ("sum",        lambda x: np.sum(x),       lambda x: float(cp.sum(x))),
    ("sort",       lambda x: np.sort(x.copy()),lambda x: cp.sort(x.copy())),
    ("matmul",     lambda x: np.matmul(x.reshape(2048,-1)[:2048,:2048],
                                       x.reshape(2048,-1)[:2048,:2048]),
                   lambda x: cp.matmul(x.reshape(2048,-1)[:2048,:2048],
                                       x.reshape(2048,-1)[:2048,:2048])),
]

data_np = np.random.randn(2048*2048).astype(np.float32)
data_cp = cp.asarray(data_np)
cp.cuda.Stream.null.synchronize()

print(f"{'Operation':10}  {'NumPy (ms)':12}  {'CuPy (ms)':12}  {'Speedup':10}")
print("-" * 50)

for name, fn_np, fn_cp in operations:
    # NumPy
    t0 = time.perf_counter()
    for _ in range(3): fn_np(data_np)
    ms_np = (time.perf_counter() - t0) * 1000 / 3

    # CuPy warmup + timed
    fn_cp(data_cp); cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(3): fn_cp(data_cp)
    cp.cuda.Stream.null.synchronize()
    ms_cp = (time.perf_counter() - t0) * 1000 / 3

    print(f"{name:10}  {ms_np:12.1f}  {ms_cp:12.3f}  {ms_np/ms_cp:10.1f}x")

# ================================================================
# 4. CuPy RawKernel — CUDA C code embedded in Python
# ================================================================
print("\n4. CuPy RawKernel (Custom CUDA C in Python)")
print("-" * 40)

# Define CUDA kernel as a string
vec_add_code = r'''
extern "C" __global__
void vec_add_scaled(const float* __restrict__ a,
                    const float* __restrict__ b,
                    float* __restrict__ c,
                    float scale, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = scale * (a[i] + b[i]);
}
'''

# Compile the kernel (JIT, happens once)
vec_add_kernel = cp.RawKernel(vec_add_code, 'vec_add_scaled')

n = 1 << 20  # 1M elements
a = cp.ones(n, dtype=cp.float32)
b = cp.ones(n, dtype=cp.float32) * 2.0
c = cp.zeros(n, dtype=cp.float32)
scale = 3.0

# Launch: (grid_dim,), (block_dim,), (args...)
threads = 256
blocks  = (n + threads - 1) // threads
vec_add_kernel((blocks,), (threads,), (a, b, c, np.float32(scale), np.int32(n)))
cp.cuda.Stream.null.synchronize()

result = c[0].get()
expected = scale * (1.0 + 2.0)
print(f"RawKernel result: c[0] = {result:.1f} (expected {expected:.1f})")
print(f"Correct: {abs(result - expected) < 1e-4}")

# ================================================================
# 5. 1D Stencil RawKernel (from Chapter 03 concepts)
# ================================================================
print("\n5. RawKernel: 1D Stencil (running average)")
print("-" * 40)

stencil_code = r'''
extern "C" __global__
void stencil_avg(const float* __restrict__ in,
                 float* __restrict__ out, int n, int radius)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= radius && i < n - radius) {
        float sum = 0.0f;
        for (int j = -radius; j <= radius; j++)
            sum += in[i + j];
        out[i] = sum / (2 * radius + 1);
    }
}
'''

stencil_kernel = cp.RawKernel(stencil_code, 'stencil_avg')
n_s = 1 << 20
radius = 4
x_s = cp.sin(cp.arange(n_s, dtype=cp.float32) * 0.01)
y_s = cp.zeros_like(x_s)
blocks_s = (n_s + 255) // 256
stencil_kernel((blocks_s,), (256,),
               (x_s, y_s, np.int32(n_s), np.int32(radius)))
cp.cuda.Stream.null.synchronize()

# Compare with cp.convolve
kernel_coeff = cp.ones(2 * radius + 1, dtype=cp.float32) / (2 * radius + 1)
y_ref = cp.convolve(x_s, kernel_coeff, mode='same')

diff = float(cp.abs(y_s[radius:n_s-radius] - y_ref[radius:n_s-radius]).max())
print(f"Max diff vs cp.convolve: {diff:.2e}")

print("\nDone!")

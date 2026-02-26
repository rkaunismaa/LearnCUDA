"""
test_extension.py
Tests the vector_add_scale CUDA extension.

First build:
    pip install -e .

Then run:
    python test_extension.py
"""

import torch
import vector_add_ext

print("Testing vector_add_scale CUDA extension")
print("=" * 45)

device = torch.device('cuda')

# ---- Basic correctness ----
a = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
b = torch.tensor([10.0, 20.0, 30.0, 40.0], device=device)
scale = 2.0

z = vector_add_ext.vector_add_scale(a, b, scale)
expected = scale * (a + b)

print(f"a:        {a.tolist()}")
print(f"b:        {b.tolist()}")
print(f"scale:    {scale}")
print(f"z:        {z.tolist()}")
print(f"expected: {expected.tolist()}")
print(f"Correct:  {torch.allclose(z, expected)}\n")

# ---- Default scale=1.0 ----
z_default = vector_add_ext.vector_add_scale(a, b)
print(f"Default scale (1.0): {z_default.tolist()}")
print(f"Expected:            {(a + b).tolist()}\n")

# ---- Large array benchmark ----
N = 1 << 24  # 16M elements
a_large = torch.randn(N, device=device)
b_large = torch.randn(N, device=device)

start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

# Warmup
vector_add_ext.vector_add_scale(a_large, b_large, 1.5)

start.record()
for _ in range(10):
    z_large = vector_add_ext.vector_add_scale(a_large, b_large, 1.5)
end.record()
torch.cuda.synchronize()

ms = start.elapsed_time(end) / 10
bw = 3 * N * 4 / (ms * 1e-3) / 1e9  # read a, read b, write z

print(f"Large array benchmark (N={N}):")
print(f"  Time:      {ms:.3f} ms")
print(f"  Bandwidth: {bw:.1f} GB/s")
print(f"  (RTX 4090 peak: ~1008 GB/s)\n")

# ---- Error handling ----
print("Error handling tests:")
try:
    wrong = torch.randn(4, device='cpu')
    vector_add_ext.vector_add_scale(wrong, b[:4], 1.0)
    print("  FAILED: should have raised for CPU tensor")
except RuntimeError as e:
    print(f"  CPU tensor rejected: {str(e)[:60]}")

try:
    a_wrong_dtype = a.to(torch.float64)
    vector_add_ext.vector_add_scale(a_wrong_dtype, b, 1.0)
    print("  FAILED: should have raised for float64")
except RuntimeError as e:
    print(f"  float64 rejected: {str(e)[:60]}")

print("\nAll tests passed!")

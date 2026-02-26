"""
Chapter 11 — 01_torch_cuda_basics.py

Demonstrates PyTorch CUDA operations:
- Device management and tensor creation
- GPU vs CPU matmul benchmark
- Accurate timing with CUDA events
- Memory monitoring
- Mixed precision (FP32 vs FP16 vs BF16)

Run:
    source /home/rob/PythonEnvironments/LearnCUDA/.learncuda/bin/activate
    python 01_torch_cuda_basics.py
"""

import torch
import time
import math

# ================================================================
# 1. Device Information
# ================================================================
print("=" * 60)
print("1. CUDA Device Information")
print("=" * 60)
print(f"CUDA available:   {torch.cuda.is_available()}")
print(f"Device count:     {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name}")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  Total Memory:       {props.total_memory / 1e9:.1f} GB")
    print(f"  Multiprocessors:    {props.multi_processor_count}")
print()

device = torch.device('cuda:0')  # Use primary GPU

# ================================================================
# 2. Tensor Creation and Movement
# ================================================================
print("=" * 60)
print("2. Tensor Creation and Movement")
print("=" * 60)

# Create directly on GPU
x_gpu = torch.zeros(1000, device=device)
y_gpu = torch.randn(1000, device=device)
print(f"x_gpu device: {x_gpu.device}, dtype: {x_gpu.dtype}")

# Move from CPU to GPU
x_cpu = torch.randn(1000)
x_gpu2 = x_cpu.to(device)
print(f"Moved to GPU: {x_gpu2.device}")

# Move back to CPU
x_back = x_gpu2.cpu()
print(f"Moved to CPU: {x_back.device}")

# Verify roundtrip
print(f"Max diff after roundtrip: {(x_back - x_cpu).abs().max().item():.2e}\n")

# ================================================================
# 3. Timing: WRONG vs CORRECT
# ================================================================
print("=" * 60)
print("3. Timing GPU Operations")
print("=" * 60)

A = torch.randn(2048, 2048, device=device)
B = torch.randn(2048, 2048, device=device)
torch.cuda.synchronize()  # Warm up

# WRONG: wall clock without synchronization
t0 = time.perf_counter()
C = torch.matmul(A, B)
t1 = time.perf_counter()
print(f"Wall clock (WRONG — async): {(t1 - t0) * 1000:.3f} ms")

# CORRECT: wall clock with synchronization
torch.cuda.synchronize()
t0 = time.perf_counter()
C = torch.matmul(A, B)
torch.cuda.synchronize()
t1 = time.perf_counter()
ms_wall = (t1 - t0) * 1000
print(f"Wall clock (sync'd):         {ms_wall:.3f} ms")

# BEST: CUDA events
start_event = torch.cuda.Event(enable_timing=True)
end_event   = torch.cuda.Event(enable_timing=True)
start_event.record()
C = torch.matmul(A, B)
end_event.record()
torch.cuda.synchronize()
ms_event = start_event.elapsed_time(end_event)
print(f"CUDA Events:                 {ms_event:.3f} ms\n")

# ================================================================
# 4. Matmul Benchmark: CPU vs GPU vs FP16
# ================================================================
print("=" * 60)
print("4. Matrix Multiply Benchmark")
print("=" * 60)
print(f"{'Size':>6}  {'CPU (ms)':>10}  {'GPU FP32':>10}  {'GPU FP16':>10}  "
      f"{'FP32 GFLOPS':>12}  {'FP16 GFLOPS':>12}")
print("-" * 72)

for N in [512, 1024, 2048, 4096]:
    flops = 2.0 * N**3

    # CPU
    Ac = torch.randn(N, N)
    Bc = torch.randn(N, N)
    t0 = time.perf_counter()
    Cc = torch.matmul(Ac, Bc)
    t1 = time.perf_counter()
    ms_cpu = (t1 - t0) * 1000

    # GPU FP32
    Ag = torch.randn(N, N, device=device)
    Bg = torch.randn(N, N, device=device)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(5): Cg = torch.matmul(Ag, Bg)
    e.record()
    torch.cuda.synchronize()
    ms_fp32 = s.elapsed_time(e) / 5

    # GPU FP16
    Ah = Ag.half()
    Bh = Bg.half()
    torch.cuda.synchronize()
    s2, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s2.record()
    for _ in range(5): Ch = torch.matmul(Ah, Bh)
    e2.record()
    torch.cuda.synchronize()
    ms_fp16 = s2.elapsed_time(e2) / 5

    gflops_fp32 = flops / (ms_fp32 * 1e-3) / 1e9
    gflops_fp16 = flops / (ms_fp16 * 1e-3) / 1e9

    print(f"{N:>6}  {ms_cpu:>10.1f}  {ms_fp32:>10.3f}  {ms_fp16:>10.3f}  "
          f"{gflops_fp32:>12.0f}  {gflops_fp16:>12.0f}")

print("\nRTX 4090 theoretical peaks:")
print("  FP32 (CUDA cores): ~82,600 GFLOPS")
print("  FP16 (Tensor Core): ~330,000 GFLOPS")

# ================================================================
# 5. Memory Management
# ================================================================
print("\n" + "=" * 60)
print("5. Memory Management")
print("=" * 60)

torch.cuda.reset_peak_memory_stats()
initial = torch.cuda.memory_allocated()

tensors = []
for i in range(5):
    t = torch.randn(1024, 1024, device=device)  # 4 MB each
    tensors.append(t)
    print(f"  After alloc #{i+1}: allocated={torch.cuda.memory_allocated()/1e6:.1f} MB")

print(f"\nPeak allocated: {torch.cuda.max_memory_allocated()/1e6:.1f} MB")
del tensors
torch.cuda.empty_cache()
print(f"After del + empty_cache: allocated={torch.cuda.memory_allocated()/1e6:.1f} MB")

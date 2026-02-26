NVCC    = nvcc
ARCH    = -arch=sm_89
CFLAGS  = -O2 -lineinfo $(ARCH) -std=c++17

TARGETS = cublas_gemm thrust_basics curand_demo

all: $(TARGETS)

cublas_gemm: 01_cublas_gemm.cu
	$(NVCC) $(CFLAGS) -o $@ $< -lcublas

thrust_basics: 02_thrust_basics.cu
	$(NVCC) $(CFLAGS) -o $@ $<

curand_demo: 03_curand.cu
	$(NVCC) $(CFLAGS) -o $@ $< -lcurand

clean:
	rm -f $(TARGETS)

.PHONY: all clean

NVCC    = nvcc
ARCH    = -arch=sm_89
CFLAGS  = -O2 -lineinfo $(ARCH) -std=c++17

TARGETS = multi_gpu vec_loads tensor_cores

all: $(TARGETS)

multi_gpu: 01_multi_gpu.cu
	$(NVCC) $(CFLAGS) -o $@ $<

vec_loads: 02_vectorized_loads.cu
	$(NVCC) $(CFLAGS) -o $@ $<

tensor_cores: 03_tensor_cores.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean

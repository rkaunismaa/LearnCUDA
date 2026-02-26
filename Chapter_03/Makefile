NVCC    = nvcc
ARCH    = -arch=sm_89
CFLAGS  = -O2 -lineinfo $(ARCH) -std=c++17

TARGETS = global_memory shared_mem const_mem

all: $(TARGETS)

global_memory: 01_global_memory.cu
	$(NVCC) $(CFLAGS) -o $@ $<

shared_mem: 02_shared_memory.cu
	$(NVCC) $(CFLAGS) -o $@ $<

const_mem: 03_constant_memory.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean

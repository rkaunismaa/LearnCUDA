NVCC    = nvcc
ARCH    = -arch=sm_89
CFLAGS  = -O2 -lineinfo $(ARCH) -std=c++17

TARGETS = reduction_naive reduction_opt atomics

all: $(TARGETS)

reduction_naive: 01_reduction_naive.cu
	$(NVCC) $(CFLAGS) -o $@ $<

reduction_opt: 02_reduction_optimized.cu
	$(NVCC) $(CFLAGS) -o $@ $<

atomics: 03_atomics.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean

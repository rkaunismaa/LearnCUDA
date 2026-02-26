NVCC    = nvcc
ARCH    = -arch=sm_89
CFLAGS  = -O2 -lineinfo $(ARCH) -std=c++17

TARGETS = pinned_memory unified_memory

all: $(TARGETS)

pinned_memory: 01_pinned_memory.cu
	$(NVCC) $(CFLAGS) -o $@ $<

unified_memory: 02_unified_memory.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean

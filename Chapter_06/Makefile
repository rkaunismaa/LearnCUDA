NVCC    = nvcc
ARCH    = -arch=sm_89
CFLAGS  = -O2 -lineinfo $(ARCH) -std=c++17

TARGETS = streams_basic async_overlap

all: $(TARGETS)

streams_basic: 01_streams_basic.cu
	$(NVCC) $(CFLAGS) -o $@ $<

async_overlap: 02_async_overlap.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean

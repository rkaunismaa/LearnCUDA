NVCC    = nvcc
ARCH    = -arch=sm_89
CFLAGS  = -O2 -lineinfo $(ARCH) -std=c++17
# Add --ptxas-options=-v to see register usage per kernel
PTXAS   = --ptxas-options=-v

TARGETS = cuda_events occupancy

all: $(TARGETS)

cuda_events: 01_cuda_events.cu
	$(NVCC) $(CFLAGS) -o $@ $<

occupancy: 02_occupancy.cu
	$(NVCC) $(CFLAGS) $(PTXAS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean

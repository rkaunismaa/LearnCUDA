NVCC    = nvcc
ARCH    = -arch=sm_89
CFLAGS  = -O2 -lineinfo $(ARCH) -std=c++17

TARGETS = warp_div warp_shuffle coop_groups

all: $(TARGETS)

warp_div: 01_warp_divergence.cu
	$(NVCC) $(CFLAGS) -o $@ $<

warp_shuffle: 02_warp_shuffle.cu
	$(NVCC) $(CFLAGS) -o $@ $<

coop_groups: 03_cooperative_groups.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean

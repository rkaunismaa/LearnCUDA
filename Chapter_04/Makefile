NVCC    = nvcc
ARCH    = -arch=sm_89
CFLAGS  = -O2 -lineinfo $(ARCH) -std=c++17

TARGETS = matmul_naive matmul_tiled

all: $(TARGETS)

matmul_naive: 01_matmul_naive.cu
	$(NVCC) $(CFLAGS) -o $@ $<

matmul_tiled: 02_matmul_tiled.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean

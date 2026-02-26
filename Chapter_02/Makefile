NVCC    = nvcc
ARCH    = -arch=sm_89
CFLAGS  = -O2 -lineinfo $(ARCH)

TARGETS = vec_add indexing

all: $(TARGETS)

vec_add: 01_vector_add.cu
	$(NVCC) $(CFLAGS) -o $@ $<

indexing: 02_thread_indexing.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean

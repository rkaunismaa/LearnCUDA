NVCC    = nvcc
ARCH    = -arch=sm_89   # RTX 4090 (Ada Lovelace). Change to sm_61 for GTX 1050.
CFLAGS  = -O2 -lineinfo $(ARCH)

TARGETS = hello device_info

all: $(TARGETS)

hello: 01_hello_cuda.cu
	$(NVCC) $(CFLAGS) -o $@ $<

device_info: 02_device_info.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean

/*
 * Chapter 01 — 02_device_info.cu
 *
 * Queries and prints detailed GPU hardware properties.
 * Understanding these numbers is essential for writing efficient CUDA code.
 *
 * Compile:
 *   nvcc -o device_info 02_device_info.cu
 * Run:
 *   ./device_info
 */

#include <stdio.h>

// Helper macro for CUDA error checking (used throughout all chapters)
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA Error at %s:%d — %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_err));          \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main()
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable GPUs found!\n");
        return 1;
    }

    printf("Found %d CUDA-capable device(s):\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        printf("=======================================================\n");
        printf("Device %d: %s\n", dev, prop.name);
        printf("=======================================================\n");

        // --- Compute Capability ---
        printf("\n[Compute Capability]\n");
        printf("  Compute Capability:        %d.%d\n", prop.major, prop.minor);

        // --- Core Counts ---
        printf("\n[Processing Units]\n");
        printf("  Multiprocessors (SMs):     %d\n", prop.multiProcessorCount);
        // CUDA cores per SM depends on compute capability
        int coresPerSM = -1;
        if      (prop.major == 9) coresPerSM = 128;  // Ada Lovelace
        else if (prop.major == 8) coresPerSM = (prop.minor == 0) ? 64 : 128; // Ampere
        else if (prop.major == 7) coresPerSM = (prop.minor == 5) ? 64 : 64;  // Turing/Volta
        else if (prop.major == 6) coresPerSM = (prop.minor == 1 || prop.minor == 2) ? 128 : 64; // Pascal
        else if (prop.major == 5) coresPerSM = 128;  // Maxwell
        if (coresPerSM > 0)
            printf("  CUDA Cores per SM:         %d\n", coresPerSM);
        if (coresPerSM > 0)
            printf("  Total CUDA Cores:          %d\n", coresPerSM * prop.multiProcessorCount);

        // --- Thread Hierarchy Limits ---
        printf("\n[Thread Hierarchy Limits]\n");
        printf("  Max threads per block:     %d\n", prop.maxThreadsPerBlock);
        printf("  Max threads per SM:        %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Max blocks per SM:         %d\n", prop.maxBlocksPerMultiProcessor);
        printf("  Warp size:                 %d\n", prop.warpSize);
        printf("  Max block dims (x,y,z):    (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max grid dims (x,y,z):     (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

        // --- Memory ---
        printf("\n[Memory]\n");
        printf("  Global memory:             %.1f GB\n",
               (double)prop.totalGlobalMem / (1 << 30));
        printf("  Shared mem per block:      %zu KB\n",
               prop.sharedMemPerBlock / 1024);
        printf("  Shared mem per SM:         %zu KB\n",
               prop.sharedMemPerMultiprocessor / 1024);
        printf("  L2 cache size:             %d MB\n",
               prop.l2CacheSize / (1 << 20));
        printf("  Registers per block:       %d\n", prop.regsPerBlock);
        printf("  Registers per SM:          %d\n", prop.regsPerMultiprocessor);
        printf("  Memory bus width:          %d bits\n", prop.memoryBusWidth);
        printf("  Memory clock rate:         %.1f MHz\n",
               prop.memoryClockRate / 1000.0f);
        // Peak memory bandwidth: 2 * clock_rate * bus_width / 8 / 1e6 GB/s
        double bw = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9;
        printf("  Peak memory bandwidth:     %.1f GB/s\n", bw);
        printf("  ECC enabled:               %s\n", prop.ECCEnabled ? "Yes" : "No");

        // --- Misc Features ---
        printf("\n[Features]\n");
        printf("  Unified addressing:        %s\n",
               prop.unifiedAddressing ? "Yes" : "No");
        printf("  Managed memory:            %s\n",
               prop.managedMemory ? "Yes" : "No");
        printf("  Concurrent kernels:        %s\n",
               prop.concurrentKernels ? "Yes" : "No");
        printf("  Async engine count:        %d\n", prop.asyncEngineCount);
        printf("  Concurrent copy+compute:   %s\n",
               prop.deviceOverlap ? "Yes" : "No");
        printf("  Peer-to-peer (NVLink):     %s\n",
               prop.directManagedMemAccessFromHost ? "Yes" : "No");
        printf("  TCC driver mode:           %s\n",
               prop.tccDriver ? "Yes" : "No");

        // --- Clock ---
        printf("\n[Clock]\n");
        printf("  GPU clock rate:            %.1f MHz\n",
               prop.clockRate / 1000.0f);

        printf("\n");
    }

    return 0;
}

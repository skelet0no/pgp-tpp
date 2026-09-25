#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int count;
    cudaGetDeviceCount(&count);

    printf("Found %d devices\n", count);

    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("Device %d\n", i);
        printf("Compute capability      : %d.%d\n",
               prop.major, prop.minor);
        printf("Name                    : %s\n", prop.name);
        printf("Total Global Memory     : %zu\n",
               prop.totalGlobalMem);
        printf("Shared memory per block : %zu\n",
               prop.sharedMemPerBlock);
        printf("Registers per block     : %d\n",
               prop.regsPerBlock);
        printf("Warp size               : %d\n",
               prop.warpSize);

        printf("Max threads per block   : (%d, %d, %d)\n",
               prop.maxThreadsDim[0],
               prop.maxThreadsDim[1],
               prop.maxThreadsDim[2]);

        printf("Max block               : (%d, %d, %d)\n",
               prop.maxGridSize[0],
               prop.maxGridSize[1],
               prop.maxGridSize[2]);

        printf("Total constant memory   : %zu\n",
               prop.totalConstMem);

        printf("Multiprocessors count   : %d\n",
               prop.multiProcessorCount);
    }

    return 0;
}
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio> // For printf in error checking

// Helper macro for checking CUDA error codes
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code %d (%s)\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CUDA Kernel: Performs element-wise addition C = A + B
__global__ void addKernel(int *c, const int *a, const int *b)
{
    // Simple 1D grid, index is just threadIdx.x
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Host function callable from main.cpp
// extern "C" ensures this function name is not C++ name-mangled.
extern "C" void runCudaAdd()
{
    const int N = 5;

    // Host input and output arrays
    int a[N] = {1, 2, 3, 4, 5};
    int b[N] = {10, 20, 30, 40, 50};
    int c[N] = {0}; // Result array

    // Device pointers
    int *d_a, *d_b, *d_c;
    size_t size = N * sizeof(int);

    // 1. Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_a, size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size));
    CUDA_CHECK(cudaMalloc((void**)&d_c, size));

    // 2. Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

    // 3. Launch the kernel: 1 block, N threads per block
    addKernel<<<1, N>>>(d_c, d_a, d_b);

    // Synchronize to ensure the kernel is finished before reading back
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. Copy results from device back to host
    CUDA_CHECK(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

    // 5. Print results and check for errors
    for (int i = 0; i < N; i++) {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }

    // 6. Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

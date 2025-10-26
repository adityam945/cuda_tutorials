#include <cuda_runtime.h>
#include <iostream>

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

extern "C" void runCudaAdd()
{
    int a[5] = {1,2,3,4,5};
    int b[5] = {10,20,30,40,50};
    int c[5] = {0};

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, 5 * sizeof(int));
    cudaMalloc(&d_b, 5 * sizeof(int));
    cudaMalloc(&d_c, 5 * sizeof(int));

    cudaMemcpy(d_a, a, 5 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, 5 * sizeof(int), cudaMemcpyHostToDevice);

    addKernel<<<1, 5>>>(d_c, d_a, d_b);

    cudaMemcpy(c, d_c, 5 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; i++)
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

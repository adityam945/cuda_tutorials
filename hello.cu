#include <iostream>

__global__ void helloFromGPU() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    helloFromGPU<<<1, 5>>>();  // 1 block, 5 threads
    cudaDeviceSynchronize();   // Wait for GPU to finish
    std::cout << "Hello from CPU!\n";
    return 0;
}

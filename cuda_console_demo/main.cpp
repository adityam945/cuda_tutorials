#include <iostream>

// Declare the function defined in the CUDA (.cu) file.
// We use extern "C" because the function in kernels.cu is defined with
// extern "C" to prevent C++ name mangling, allowing the linker to find it easily.
extern "C" void runCudaAdd();

int main() {
    std::cout << "Running CUDA addition demo..." << std::endl;
    runCudaAdd();
    std::cout << "Demo complete." << std::endl;
    return 0;
}

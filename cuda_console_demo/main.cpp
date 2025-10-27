#include <iostream>
#include <thread>
#include <chrono>
#include "worker/worker_api.h" // NEW: Access worker functions

// Declare the function defined in the CUDA (.cu) file.
extern "C" void runCudaAdd();

int main() {
    // --- 1. Run the CUDA Demo ---
    std::cout << "--- 1. Running CUDA Addition Demo (Synchronous) ---" << std::endl;
    runCudaAdd();
    std::cout << "CUDA Demo complete." << std::endl;

    // --- 2. Launch the Worker Threads ---
    std::cout << "\n--- 2. Launching Worker Threads (Asynchronous) ---" << std::endl;
    runWorkerDemo();

    // --- 3. Keep Main Thread Alive ---
    std::cout << "Main application running. Threads started in background. Press Ctrl+C to stop." << std::endl;
    
    while(true) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        // Use the new public function to report status
        std::cout << "MAIN: Running in background (Queue size: " << getQueueSize() << ")" << std::endl;
    }
    
    // The program will exit via Ctrl+C, not return 0.
}

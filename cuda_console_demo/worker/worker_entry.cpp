#include "worker_api.h" // Use extern declarations
#include <iostream>
#include <thread>
#include <chrono>

using namespace std::chrono;

// --- Producer Function (Internal - static/anonymous namespace not strictly needed) ---
void producer_thread_func() {
    std::cout << "PRODUCER: Starting continuous work (25 items/sec)..." << std::endl;
    int produced_count = 0;
    
    const double interval_ms = 1000.0 / 25.0;
    const milliseconds interval_per_item(static_cast<long long>(interval_ms));
    
    auto start_time = steady_clock::now();
    int i = 0;

    while (true) { 
        auto loop_start = steady_clock::now();
        
        std::string message = "Message #" + std::to_string(i++);
        shared_queue.push(message); // Uses the shared_queue defined in queue.cpp
        produced_count++;

        if (produced_count % 25 == 0) {
            auto elapsed_ms = duration_cast<milliseconds>(steady_clock::now() - start_time).count();
            std::cout << "PRODUCER: Added 25 items. Total: " << produced_count 
                      << " (Queue size: " << shared_queue.size() << ")" 
                      << " Avg rate: " << (double)produced_count / (elapsed_ms / 1000.0) << " items/s" << std::endl;
        }

        auto time_taken = steady_clock::now() - loop_start;
        auto sleep_duration = interval_per_item - time_taken;
        
        if (sleep_duration > milliseconds(0)) {
            std::this_thread::sleep_for(sleep_duration);
        }
    }
}

// --- Consumer Function (Internal) ---
void consumer_thread_func() {
    std::cout << "CONSUMER: Starting continuous work..." << std::endl;
    int consumed_count = 0;
    
    while (true) { 
        WorkItem item = shared_queue.pop(); 
        
        consumed_count++;
        
        if (consumed_count % 100 == 0) {
             std::cout << "CONSUMER: Processed " << consumed_count << " items. Queue size: " << shared_queue.size() << std::endl;
        }
    }
}

// Public entry point for the worker logic
void runWorkerDemo() {
    std::cout << "Starting Continuous Worker Demo (25 items/sec producer rate)" << std::endl;
    
    std::thread producer(producer_thread_func);
    std::thread consumer(consumer_thread_func);
    
    producer.detach();
    consumer.detach();
}

#include "queue.h"
#include "worker_api.h" // Include the API header
#include <iostream>

// --- Queue Implementation (Unchanged) ---
void WorkQueue::push(const WorkItem& item) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(item);
    } 
    cond_var_.notify_one();
}

WorkItem WorkQueue::pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_var_.wait(lock, [this]{ return !queue_.empty(); });
    WorkItem item = queue_.front();
    queue_.pop();
    return item;
}

bool WorkQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

size_t WorkQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

// --- GLOBAL DEFINITION (ODR FIX) ---
// DEFINE the global queue variable here, and ONLY here.
WorkQueue shared_queue;

// Helper function definition
size_t getQueueSize() {
    return shared_queue.size();
}

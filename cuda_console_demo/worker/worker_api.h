#ifndef WORKER_API_H
#define WORKER_API_H

#include "queue.h"

// Declare the global shared queue (defined once in worker/queue.cpp)
extern WorkQueue shared_queue;

// Declare the worker entry point function
void runWorkerDemo();

// Declare helper function for accessing queue status
size_t getQueueSize();

#endif // WORKER_API_H

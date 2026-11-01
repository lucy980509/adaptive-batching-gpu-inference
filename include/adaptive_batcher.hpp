#pragma once

#include "request_queue.hpp"

#include <chrono>
#include <cstddef>
#include <vector>

std::vector<Request> collect_adaptive_batch(
    RequestQueue& request_queue,
    std::size_t max_batch_size,
    std::chrono::milliseconds max_wait
);

std::vector<Request> collect_adaptive_immediate_batch(
    RequestQueue& request_queue,
    std::size_t max_batch_size
);
#pragma once

#include "request_queue.hpp"

#include <vector>

std::vector<Request> collect_static_batch(
    RequestQueue& request_queue,
    int batch_size
);
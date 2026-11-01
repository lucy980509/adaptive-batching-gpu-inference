#pragma once

#include "request_queue.hpp"
#include "workload.hpp"

#include <vector>

void generate_requests(
    RequestQueue& request_queue,
    const std::vector<float>& input_data,
    const WorkloadConfig& config
);
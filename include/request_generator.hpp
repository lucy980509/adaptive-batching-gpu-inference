#pragma once

#include "request_queue.hpp"
#include "workload.hpp"

#include <vector>

// Cost of preparing one request after its arrival time is stamped: copying the
// image into Request.input and pushing it (another copy) into the queue.
// Both happen inside the request's latency budget and can delay the next arrival.
struct GeneratorStats {
    uint64_t scheduled = 0;  // arrival slots planned by the workload
    uint64_t generated = 0;  // requests actually created; the rest were not
                             // generated because the window had already ended
    double prepare_mean_us = 0.0;
    double prepare_max_us = 0.0;
    TimePoint start{};
    TimePoint end{};  // configured end of the generation window
};

// Request k (ids start at 1) always carries image (k - 1) % pool size, so every
// policy run with the same pool sees the same image for the same request id.
GeneratorStats generate_requests(
    RequestQueue& request_queue,
    const std::vector<std::vector<float>>& image_pool,
    const WorkloadConfig& config,
    std::chrono::milliseconds latency_target = SloConfig{}.latency_target
);

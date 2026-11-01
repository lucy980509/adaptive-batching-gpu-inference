#pragma once

#include "request_queue.hpp"
#include "service_time_estimator.hpp"
#include "slo_config.hpp"

#include <cstddef>
#include <vector>

std::vector<Request> collect_slo_batch(
    RequestQueue& request_queue,
    std::size_t max_batch_size,
    const ServiceTimeEstimator& estimator,
    const SloConfig& config
);

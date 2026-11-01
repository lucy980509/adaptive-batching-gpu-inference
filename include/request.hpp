#pragma once

#include "slo_config.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

struct Request {
    uint64_t id;
    TimePoint arrival_time;
    TimePoint scheduled_arrival_time;
    TimePoint deadline;

    std::vector<float> input;
    TimePoint start_service_time{};
    TimePoint finish_time{};
    double estimated_service_ms = 0.0;
    std::size_t image_index = 0;  // which image of the pool this request carries

    Request(
        uint64_t request_id,
        std::chrono::milliseconds latency_target = SloConfig{}.latency_target
    ) {
        id = request_id;
        arrival_time = Clock::now();
        scheduled_arrival_time = arrival_time;
        deadline = arrival_time + latency_target;
    }
};

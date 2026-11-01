#pragma once

#include <chrono>

struct SloConfig {
    std::chrono::milliseconds latency_target{20};
    std::chrono::milliseconds safety_margin{10};
    double target_attainment_percent = 95.0;
};

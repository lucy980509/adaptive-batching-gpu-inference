#pragma once

#include <chrono>

// Defaults for the ResNet-50 pilot. These are the only place the default SLO
// numbers live; p3_inference can override them with --slo-ms, --margin-ms and
// --required-percent, and every request row records the values actually used.
struct SloConfig {
    std::chrono::milliseconds latency_target{100};
    std::chrono::milliseconds safety_margin{10};
    double target_attainment_percent = 95.0;
};

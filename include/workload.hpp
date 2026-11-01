#pragma once

#include <string>

struct WorkloadConfig {
    std::string name;
    double requests_per_second;
    int duration_seconds;

    int burst_on_ms;
    int burst_off_ms;
};
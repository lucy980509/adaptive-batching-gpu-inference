#pragma once

#include <stdexcept>
#include <string>

struct WorkloadConfig {
    std::string name;
    double requests_per_second;
    int duration_seconds;

    int burst_on_ms;
    int burst_off_ms;
};

const int kDefaultWorkloadSeconds = 10;

// ResNet-50 pilot workloads. This table is the only place the request rates
// live; the rates and the duration actually used are written to every CSV row.
// Bursty: requests_per_second applies only during the "on" window.
inline WorkloadConfig get_workload(const std::string& name, int duration_seconds) {
    if (name == "low") {
        return {"low", 20.0, duration_seconds, 0, 0};
    }
    if (name == "medium") {
        return {"medium", 100.0, duration_seconds, 0, 0};
    }
    if (name == "high") {
        return {"high", 250.0, duration_seconds, 0, 0};
    }
    if (name == "bursty") {
        return {"bursty", 250.0, duration_seconds, 500, 1500};
    }

    throw std::invalid_argument("Unknown workload: " + name);
}

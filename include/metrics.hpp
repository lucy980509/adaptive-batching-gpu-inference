#pragma once

#include "request.hpp"
#include "workload.hpp"

#include <fstream>
#include <string>
#include <vector>

class MetricsLogger {
private:
    std::ofstream metrics_file;

    WorkloadConfig config;
    int batch_size;
    std::string mode;
    SloConfig slo_config;

    TimePoint first_arrival{};
    TimePoint last_finish{};
    int completed_requests = 0;
    int slo_met_requests = 0;

public:
    MetricsLogger(
        const std::string& csv_path,
        const WorkloadConfig& workload_config,
        int configured_batch_size,
        const std::string& policy_mode = "unknown",
        const SloConfig& target_config = SloConfig{}
    );

    bool is_open() const;

    void record_batch(const std::vector<Request>& batch);

    void print_summary() const;
};

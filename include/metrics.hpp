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
    std::string model_name;
    int max_wait_ms;

    TimePoint first_arrival{};
    TimePoint first_scheduled{};
    TimePoint last_finish{};
    int completed_requests = 0;
    int slo_met_requests = 0;

public:
    MetricsLogger(
        const std::string& csv_path,
        const WorkloadConfig& workload_config,
        int configured_batch_size,
        const std::string& policy_mode = "unknown",
        const SloConfig& target_config = SloConfig{},
        const std::string& model = "resnet50",
        int configured_max_wait_ms = -1  // -1: not applicable to this policy
    );

    bool is_open() const;

    TimePoint first_arrival_time() const { return first_arrival; }
    TimePoint last_finish_time() const { return last_finish; }

    void record_batch(const std::vector<Request>& batch);

    void print_summary() const;
};

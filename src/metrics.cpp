#include "../include/metrics.hpp"

#include <chrono>
#include <iostream>

MetricsLogger::MetricsLogger(
    const std::string& csv_path,
    const WorkloadConfig& workload_config,
    int configured_batch_size
)
    : metrics_file(csv_path),
      config(workload_config),
      batch_size(configured_batch_size) {

    if (metrics_file.is_open()) {
        metrics_file
            << "workload,target_rps,duration_seconds,"
            << "burst_on_ms,burst_off_ms,max_batch_size,actual_batch_size,"
            << "request_id,queue_wait_ms,service_time_ms,e2e_latency_ms,"
            << "arrival_offset_ms,generation_lag_ms\n";
    }
}

bool MetricsLogger::is_open() const {
    return metrics_file.is_open();
}

void MetricsLogger::record_batch(const std::vector<Request>& batch) {
    for (const Request& request : batch) {
        if (completed_requests == 0) {
            first_arrival = request.arrival_time;
        }

        last_finish = request.finish_time;
        ++completed_requests;

        double wait_ms =
            std::chrono::duration<double, std::milli>(
                request.start_service_time - request.arrival_time
            ).count();

        double service_ms =
            std::chrono::duration<double, std::milli>(
                request.finish_time - request.start_service_time
            ).count();

        double total_ms =
            std::chrono::duration<double, std::milli>(
                request.finish_time - request.arrival_time
            ).count();

        double arrival_offset_ms =
            std::chrono::duration<double, std::milli>(
                request.arrival_time - first_arrival
            ).count();

        double generation_lag_ms =
            std::chrono::duration<double, std::milli>(
                request.arrival_time - request.scheduled_arrival_time
            ).count();

        metrics_file
            << config.name << ','
            << config.requests_per_second << ','
            << config.duration_seconds << ','
            << config.burst_on_ms << ','
            << config.burst_off_ms << ','
            << batch_size << ','
            << batch.size() << ','
            << request.id << ','
            << wait_ms << ','
            << service_ms << ','
            << total_ms << ','
            << arrival_offset_ms << ','
            << generation_lag_ms << '\n';
    }
}

void MetricsLogger::print_summary() const {
    if (completed_requests == 0) {
        std::cout << "No requests completed.\n";
        return;
    }

    double elapsed_seconds = std::chrono::duration<double>(
        last_finish - first_arrival
    ).count();

    std::cout << "Workload: " << config.name << '\n';
    std::cout << "Completed requests: "
              << completed_requests << '\n';

    if (elapsed_seconds > 0.0) {
        std::cout << "Throughput: "
                  << completed_requests / elapsed_seconds
                  << " requests/sec\n";
    }
}
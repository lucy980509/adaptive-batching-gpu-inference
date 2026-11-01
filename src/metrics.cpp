#include "../include/metrics.hpp"

#include <chrono>
#include <iostream>
#include <iomanip>

MetricsLogger::MetricsLogger(
    const std::string& csv_path,
    const WorkloadConfig& workload_config,
    int configured_batch_size,
    const std::string& policy_mode,
    const SloConfig& target_config,
    const std::string& model,
    int configured_max_wait_ms
)
    : metrics_file(csv_path),
      config(workload_config),
      batch_size(configured_batch_size),
      mode(policy_mode),
      slo_config(target_config),
      model_name(model),
      max_wait_ms(configured_max_wait_ms) {

    if (metrics_file.is_open()) {
        metrics_file << std::setprecision(12);
        metrics_file
            << "workload,target_rps,duration_seconds,"
            << "burst_on_ms,burst_off_ms,max_batch_size,actual_batch_size,"
            << "request_id,queue_wait_ms,service_time_ms,e2e_latency_ms,"
            << "arrival_offset_ms,generation_lag_ms,"
            << "mode,slo_target_ms,slo_required_percent,safety_margin_ms,"
            << "deadline_offset_ms,estimated_service_ms,"
            << "slack_at_service_start_ms,slo_met,model,"
            << "image_index,max_wait_ms,scheduled_offset_ms,"
            << "service_minus_estimate_ms\n";
    }
}

bool MetricsLogger::is_open() const {
    return metrics_file.is_open();
}

void MetricsLogger::record_batch(const std::vector<Request>& batch) {
    for (const Request& request : batch) {
        if (completed_requests == 0) {
            first_arrival = request.arrival_time;
            first_scheduled = request.scheduled_arrival_time;
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

        const double deadline_offset_ms =
            std::chrono::duration<double, std::milli>(
                request.deadline - first_arrival
            ).count();

        const double slack_ms =
            std::chrono::duration<double, std::milli>(
                request.deadline - request.start_service_time
            ).count() - request.estimated_service_ms;

        // Planned arrival relative to request 1's planned arrival. It does not
        // depend on the policy, so it can be compared between runs.
        const double scheduled_offset_ms =
            std::chrono::duration<double, std::milli>(
                request.scheduled_arrival_time - first_scheduled
            ).count();

        // Measured service time minus the calibration estimate of this batch.
        const double service_minus_estimate_ms =
            service_ms - request.estimated_service_ms;

        const bool slo_met = request.finish_time <= request.deadline;
        if (slo_met) {
            ++slo_met_requests;
        }

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
            << generation_lag_ms << ','
            << mode << ','
            << slo_config.latency_target.count() << ','
            << slo_config.target_attainment_percent << ','
            << slo_config.safety_margin.count() << ','
            << deadline_offset_ms << ','
            << request.estimated_service_ms << ','
            << slack_ms << ','
            << (slo_met ? 1 : 0) << ','
            << model_name << ','
            << request.image_index << ','
            << (max_wait_ms >= 0 ? std::to_string(max_wait_ms) : std::string())
            << ',' << scheduled_offset_ms << ','
            << service_minus_estimate_ms << '\n';
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

    const double attainment_percent =
        100.0 * slo_met_requests / completed_requests;

    std::cout << "SLO target: " << slo_config.latency_target.count()
              << " ms\n";
    std::cout << "SLO met requests: " << slo_met_requests
              << '/' << completed_requests << '\n';
    std::cout << "SLO attainment: " << attainment_percent << "%\n";
    std::cout << "SLO violations: "
              << completed_requests - slo_met_requests << '\n';
    std::cout << "SLO requirement: "
              << slo_config.target_attainment_percent << "%\n";
    std::cout << "SLO status: "
              << (attainment_percent >= slo_config.target_attainment_percent
                  ? "PASS" : "FAIL") << '\n';
}

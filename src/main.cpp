#include "../include/request_queue.hpp"
#include "../include/preprocess.hpp"
#include "../include/inference_engine.hpp"
#include "../include/workload.hpp"
#include "../include/request_generator.hpp"
#include "../include/static_batcher.hpp"
#include "../include/adaptive_batcher.hpp"
#include "../include/metrics.hpp"
#include "../include/slo_batcher.hpp"
#include "../include/slo_config.hpp"
#include "../include/service_time_estimator.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr
            << "Usage: p3_inference.exe "
            << "<workload> <static|adaptive|adaptive_immediate|slo> "
            << "<batch_size> <csv_path>\n";
        return 1;
    }

    const std::string workload_name = argv[1];
    const std::string mode = argv[2];
    const std::string batch_argument = argv[3];
    const std::string csv_path = argv[4];

    if (mode != "static" &&
        mode != "adaptive" &&
        mode != "adaptive_immediate" &&
        mode != "slo") {
        std::cerr
            << "Mode must be static, adaptive, adaptive_immediate or slo.\n";
        return 1;
    }

    if (batch_argument != "1" &&
        batch_argument != "4" &&
        batch_argument != "8" &&
        batch_argument != "16" &&
        batch_argument != "32") {
        std::cerr << "Batch size must be 1, 4, 8, 16 or 32.\n";
        return 1;
    }

    const int batch_size = std::stoi(batch_argument);
    const auto max_wait = std::chrono::milliseconds(10);
    const SloConfig slo_config{};

    WorkloadConfig config{};

    if (workload_name == "low") {
        config = {"low", 2.0, 10, 0, 0};
    }
    else if (workload_name == "medium") {
        config = {"medium", 20.0, 10, 0, 0};
    }
    else if (workload_name == "high") {
        config = {"high", 200.0, 10, 0, 0};
    }
    else if (workload_name == "bursty") {
        config = {"bursty", 200.0, 10, 500, 1500};
    }
    else {
        std::cerr << "Unknown workload.\n";
        return 1;
    }

    InferenceEngine engine(L"models/wafer_fault_cnn.onnx");

    std::vector<float> input_data =
        preprocess_image("test_images/sample.png");

    if (input_data.empty()) {
        return 1;
    }

    MetricsLogger metrics(csv_path, config, batch_size, mode, slo_config);

    if (!metrics.is_open()) {
        std::cerr << "Failed to open CSV: " << csv_path << '\n';
        return 1;
    }

    std::cout << "Warm-up started.\n";
    const int warmup_runs = 3;
    const int calibration_runs = 10;
    ServiceTimeEstimator estimator(32);

    for (int size = 1; size <= 32; ++size) {
        std::vector<Request> warmup_batch;

        for (int i = 0; i < size; ++i) {
            Request req(0);
            req.input = input_data;
            warmup_batch.push_back(req);
        }

        for (int run = 0; run < warmup_runs; ++run) {
            engine.run_batch(warmup_batch);
        }

        for (int run = 0; run < calibration_runs; ++run) {
            engine.run_batch(warmup_batch);

            const double service_ms =
                std::chrono::duration<double, std::milli>(
                    warmup_batch.front().finish_time -
                    warmup_batch.front().start_service_time
                ).count();

            estimator.observe(static_cast<std::size_t>(size), service_ms);
        }
    }

    estimator.finalize();

    const std::string profile_path = csv_path + ".service_times.csv";
    std::ofstream profile_file(profile_path);

    if (!profile_file.is_open()) {
        std::cerr << "Failed to open calibration CSV: " << profile_path << '\n';
        return 1;
    }

    profile_file << std::setprecision(12);
    profile_file
        << "batch_size,estimated_service_ms,calibration_runs,"
        << "safety_margin_ms,estimator\n";

    for (std::size_t size = 1; size <= 32; ++size) {
        profile_file << size << ',' << estimator.estimate_ms(size) << ','
                     << calibration_runs << ','
                     << slo_config.safety_margin.count() << ','
                     << "prefix_max_of_calibration_max\n";
    }

    profile_file.close();
    std::cout << "Warm-up completed.\n";
    std::cout << "Calibration saved: " << profile_path << '\n';
    std::cout << "SLO budget: " << slo_config.latency_target.count()
              << " ms | Required attainment: "
              << slo_config.target_attainment_percent
              << "% | Safety margin: "
              << slo_config.safety_margin.count() << " ms\n";

    RequestQueue request_queue;

    std::cout << "MEASURE_START" << std::endl;

    std::thread producer([&request_queue, &input_data, config, slo_config]() {
        generate_requests(
            request_queue, input_data, config, slo_config.latency_target
        );
    });

    try {
        while (true) {
            std::vector<Request> batch;

            if (mode == "static") {
                batch = collect_static_batch(
                    request_queue,
                    batch_size
                );
            }
            else if (mode == "adaptive") {
                batch = collect_adaptive_batch(
                    request_queue,
                    batch_size,
                    max_wait
                );
            }
            else if (mode == "adaptive_immediate") {
                batch = collect_adaptive_immediate_batch(
                    request_queue,
                    batch_size
                );
            }
            else {
                batch = collect_slo_batch(
                    request_queue, batch_size, estimator, slo_config
                );
            }

            if (batch.empty()) {
                break;
            }

            const double estimated_ms = estimator.estimate_ms(batch.size());

            for (Request& request : batch) {
                request.estimated_service_ms = estimated_ms;
            }

            engine.run_batch(batch);
            metrics.record_batch(batch);
        }

        std::cout << "MEASURE_END" << std::endl;
    }
    catch (...) {
        producer.join();
        throw;
    }

    producer.join();

    std::cout << "Mode: " << mode << '\n';
    std::cout << "Configured batch size: " << batch_size << '\n';

    metrics.print_summary();

    return 0;
}

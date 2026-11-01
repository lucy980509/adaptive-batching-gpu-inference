#include "../include/request_queue.hpp"
#include "../include/preprocess.hpp"
#include "../include/inference_engine.hpp"
#include "../include/workload.hpp"
#include "../include/request_generator.hpp"
#include "../include/static_batcher.hpp"
#include "../include/adaptive_batcher.hpp"
#include "../include/metrics.hpp"

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr
            << "Usage: p3_inference.exe "
            << "<workload> <static|adaptive|adaptive_immediate> "
            << "<batch_size> <csv_path>\n";
        return 1;
    }

    const std::string workload_name = argv[1];
    const std::string mode = argv[2];
    const std::string batch_argument = argv[3];
    const std::string csv_path = argv[4];

    if (mode != "static" &&
        mode != "adaptive" &&
        mode != "adaptive_immediate") {
        std::cerr
            << "Mode must be static, adaptive or adaptive_immediate.\n";
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

    MetricsLogger metrics(csv_path, config, batch_size);

    if (!metrics.is_open()) {
        std::cerr << "Failed to open CSV: " << csv_path << '\n';
        return 1;
    }

    std::cout << "Warm-up started.\n";

    for (int size = 1; size <= 32; ++size) {
        std::vector<Request> warmup_batch;

        for (int i = 0; i < size; ++i) {
            Request req(0);
            req.input = input_data;
            warmup_batch.push_back(req);
        }

        for (int run = 0; run < 3; ++run) {
            engine.run_batch(warmup_batch);
        }
    }

    std::cout << "Warm-up completed.\n";

    RequestQueue request_queue;

    std::thread producer([&request_queue, &input_data, config]() {
        generate_requests(request_queue, input_data, config);
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
            else {
                batch = collect_adaptive_immediate_batch(
                    request_queue,
                    batch_size
                );
            }

            if (batch.empty()) {
                break;
            }

            engine.run_batch(batch);
            metrics.record_batch(batch);
        }
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
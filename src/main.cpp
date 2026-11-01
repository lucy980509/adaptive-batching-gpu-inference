#include "../include/request_queue.hpp"
#include "../include/inference_engine.hpp"
#include "../include/workload.hpp"
#include "../include/request_generator.hpp"
#include "../include/static_batcher.hpp"
#include "../include/adaptive_batcher.hpp"
#include "../include/metrics.hpp"
#include "../include/slo_batcher.hpp"
#include "../include/slo_config.hpp"
#include "../include/service_time_estimator.hpp"
#include "../include/model_config.hpp"
#include "../include/dataset.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

bool is_supported_batch_size(const std::string& text) {
    return text == "1" || text == "4" || text == "8" ||
           text == "16" || text == "32";
}

// Builds `size` requests from the first `size` prepared ImageNet images.
// The images and their order are fixed, so every run sees the same input.
std::vector<Request> make_fixed_batch(
    const ModelConfig& model,
    std::size_t size
) {
    const std::vector<std::vector<float>> images = read_prepared_images(
        kPreparedImagePath, kPreparedImageCount,
        model.input_elements(), 0, size
    );

    std::vector<Request> batch;

    for (std::size_t i = 0; i < size; ++i) {
        Request req(i);
        req.input = images[i];
        batch.push_back(req);
    }

    return batch;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr
            << "Usage: p3_inference.exe "
            << "<workload> <static|adaptive|adaptive_immediate|slo> "
            << "<batch_size> <csv_path>\n"
            << "       p3_inference.exe <any> model_check <1|4> <logits_path>\n"
            << "       p3_inference.exe <any> batch_profile "
            << "<1|4|8|16|32> <csv_path>\n";
        return 1;
    }

    const std::string workload_name = argv[1];
    const std::string mode = argv[2];
    const std::string batch_argument = argv[3];
    const std::string csv_path = argv[4];

    if (mode != "static" &&
        mode != "adaptive" &&
        mode != "adaptive_immediate" &&
        mode != "slo" &&
        mode != "model_check" &&
        mode != "batch_profile") {
        std::cerr
            << "Mode must be static, adaptive, adaptive_immediate, slo, "
            << "model_check or batch_profile.\n";
        return 1;
    }

    const ModelConfig model = resnet50_model_config();

    // Model check: load the model on CUDA and run the first 1 or 4 prepared
    // images once. The logits are written as raw float32 to <logits_path> so
    // they can be compared with PyTorch (scripts/verify_resnet_inputs.py).
    // This is the only place the output is copied; the serving loop and
    // batch_profile never request it.
    if (mode == "model_check") {
        if (batch_argument != "1" && batch_argument != "4") {
            std::cerr << "model_check runs batch size 1 or 4.\n";
            return 1;
        }

        try {
            InferenceEngine engine(model);

            std::vector<Request> batch =
                make_fixed_batch(model, std::stoul(batch_argument));

            std::vector<float> output;
            engine.run_batch(batch, &output);

            const std::size_t per_image = output.size() / batch.size();
            std::cout << "MODEL_CHECK model=" << model.name
                      << " batch=" << batch.size()
                      << " output_values=" << output.size() << '\n';

            for (std::size_t n = 0; n < batch.size(); ++n) {
                std::size_t top1 = 0;
                for (std::size_t i = 1; i < per_image; ++i) {
                    if (output[n * per_image + i] >
                        output[n * per_image + top1]) {
                        top1 = i;
                    }
                }
                std::cout << "  image " << n << " top1_index=" << top1 << '\n';
            }

            std::ofstream logits_file(csv_path, std::ios::binary);
            logits_file.write(
                reinterpret_cast<const char*>(output.data()),
                static_cast<std::streamsize>(output.size() * sizeof(float))
            );

            if (!logits_file) {
                throw std::runtime_error("Cannot write logits: " + csv_path);
            }

            std::cout << "Logits written: " << csv_path << '\n'
                      << "Model check only, not a serving or performance "
                      << "result.\n";
        }
        catch (const std::exception& error) {
            std::cerr << "Model check failed: " << error.what() << '\n';
            return 1;
        }

        return 0;
    }

    // Batch profile: sequential batches of the same fixed images, no arriving
    // traffic. Model loading, image loading and 3 warm-up runs are not timed;
    // the next 10 runs are timed with the engine's own service-time clock
    // (batch assembly + tensor setup + session.Run). No logits are copied and
    // nothing is printed between runs. One batch size per process, so a
    // failure (for example out of GPU memory) cannot affect another size.
    if (mode == "batch_profile") {
        if (!is_supported_batch_size(batch_argument)) {
            std::cerr << "Batch size must be 1, 4, 8, 16 or 32.\n";
            return 1;
        }

        const int warmup_runs = 3;
        const int measured_runs = 10;

        try {
            InferenceEngine engine(model);

            std::vector<Request> batch =
                make_fixed_batch(model, std::stoul(batch_argument));

            for (int run = 0; run < warmup_runs; ++run) {
                engine.run_batch(batch);
            }

            std::vector<double> service_ms;

            for (int run = 0; run < measured_runs; ++run) {
                engine.run_batch(batch);

                service_ms.push_back(
                    std::chrono::duration<double, std::milli>(
                        batch.front().finish_time -
                        batch.front().start_service_time
                    ).count()
                );
            }

            std::ofstream profile_file(csv_path);

            if (!profile_file.is_open()) {
                throw std::runtime_error("Cannot write " + csv_path);
            }

            profile_file << std::setprecision(12);
            profile_file << "model,batch_size,run,service_ms\n";

            for (int run = 0; run < measured_runs; ++run) {
                profile_file << model.name << ',' << batch.size() << ','
                             << run + 1 << ',' << service_ms[run] << '\n';
            }

            std::cout << "BATCH_PROFILE_DONE batch=" << batch.size()
                      << " warmup=" << warmup_runs
                      << " measured=" << measured_runs << '\n';
        }
        catch (const std::exception& error) {
            std::cerr << "Batch profile failed: " << error.what() << '\n';
            return 1;
        }

        return 0;
    }

    if (!is_supported_batch_size(batch_argument)) {
        std::cerr << "Batch size must be 1, 4, 8, 16 or 32.\n";
        return 1;
    }

    // The generator still copies one input into every request. The fixed image
    // pool with one shared order for all policies is the next step, so serving
    // is switched off until then instead of reusing a single image.
    const bool image_pool_connected = false;

    if (!image_pool_connected) {
        std::cerr
            << "Serving is disabled until the ImageNet image pool is "
            << "connected to the request generator. Use model_check or "
            << "batch_profile.\n";
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

    std::unique_ptr<InferenceEngine> engine_holder;

    try {
        engine_holder = std::make_unique<InferenceEngine>(model);
    }
    catch (const std::exception& error) {
        std::cerr << "Failed to load model: " << error.what() << '\n';
        return 1;
    }

    InferenceEngine& engine = *engine_holder;

    // Placeholder until the image pool exists (unreachable while
    // image_pool_connected is false): the first prepared image.
    std::vector<float> input_data =
        read_prepared_images(
            kPreparedImagePath, kPreparedImageCount,
            model.input_elements(), 0, 1
        )[0];

    MetricsLogger metrics(
        csv_path, config, batch_size, mode, slo_config, model.name
    );

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

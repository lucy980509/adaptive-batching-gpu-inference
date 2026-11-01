#define NOMINMAX
#define PSAPI_VERSION 2
#include <windows.h>
#include <psapi.h>

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

// Peak physical memory of this process so far, in MiB (-1 if unavailable).
double peak_working_set_mib() {
    PROCESS_MEMORY_COUNTERS counters{};

    if (!GetProcessMemoryInfo(
            GetCurrentProcess(), &counters, sizeof(counters))) {
        return -1.0;
    }

    return static_cast<double>(counters.PeakWorkingSetSize) / (1024.0 * 1024.0);
}

// Optional settings after the four positional arguments. Defaults live in
// workload.hpp (rates), slo_config.hpp (SLO numbers) and here (max wait);
// anything not given on the command line keeps its default.
struct RunOptions {
    int duration_seconds = kDefaultWorkloadSeconds;
    SloConfig slo;
    int max_wait_ms = 10;
};

RunOptions parse_options(int argc, char* argv[]) {
    RunOptions options;

    for (int i = 5; i < argc; i += 2) {
        const std::string name = argv[i];

        if (i + 1 >= argc) {
            throw std::invalid_argument("Missing value for " + name);
        }

        const double value = std::stod(argv[i + 1]);

        if (name == "--duration") {
            options.duration_seconds = static_cast<int>(value);
        }
        else if (name == "--slo-ms") {
            options.slo.latency_target =
                std::chrono::milliseconds(static_cast<long long>(value));
        }
        else if (name == "--margin-ms") {
            options.slo.safety_margin =
                std::chrono::milliseconds(static_cast<long long>(value));
        }
        else if (name == "--required-percent") {
            options.slo.target_attainment_percent = value;
        }
        else if (name == "--max-wait-ms") {
            options.max_wait_ms = static_cast<int>(value);
        }
        else {
            throw std::invalid_argument("Unknown option: " + name);
        }
    }

    if (options.duration_seconds <= 0 ||
        options.slo.latency_target.count() <= 0 ||
        options.slo.safety_margin.count() < 0 ||
        options.slo.target_attainment_percent <= 0.0 ||
        options.slo.target_attainment_percent > 100.0 ||
        options.max_wait_ms < 0) {
        throw std::invalid_argument("An option value is out of range");
    }

    return options;
}

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
    if (argc < 5) {
        std::cerr
            << "Usage: p3_inference.exe "
            << "<workload> <static|adaptive|adaptive_immediate|slo> "
            << "<batch_size> <csv_path> [options]\n"
            << "       p3_inference.exe <any> model_check <1|4> <logits_path>\n"
            << "       p3_inference.exe <any> batch_profile "
            << "<1|4|8|16|32> <csv_path>\n"
            << "Options (serving): --duration <s> --slo-ms <ms> "
            << "--margin-ms <ms> --required-percent <p> --max-wait-ms <ms>\n";
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

    RunOptions options;

    try {
        options = parse_options(argc, argv);
    }
    catch (const std::exception& error) {
        std::cerr << "Bad option: " << error.what() << '\n';
        return 1;
    }

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

    // batch_size is the maximum batch size: the fixed size for Static, the
    // upper limit for the other policies.
    const int batch_size = std::stoi(batch_argument);

    WorkloadConfig config{};

    try {
        config = get_workload(workload_name, options.duration_seconds);
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << " (use low, medium, high or bursty)\n";
        return 1;
    }

    const auto max_wait = std::chrono::milliseconds(options.max_wait_ms);
    const SloConfig& slo_config = options.slo;

    std::unique_ptr<InferenceEngine> engine_holder;
    std::vector<std::vector<float>> image_pool;

    try {
        engine_holder = std::make_unique<InferenceEngine>(model);

        // The whole prepared subset is read into RAM once, before anything is
        // measured. Requests copy from it by index.
        image_pool = read_prepared_images(
            kPreparedImagePath, kPreparedImageCount,
            model.input_elements(), 0, kPreparedImageCount
        );
    }
    catch (const std::exception& error) {
        std::cerr << "Startup failed: " << error.what() << '\n';
        return 1;
    }

    InferenceEngine& engine = *engine_holder;

    std::cout << "Image pool: " << image_pool.size()
              << " images loaded; peak working set "
              << peak_working_set_mib() << " MiB\n";

    // Only Adaptive wait uses a maximum wait; Adaptive immediate waits 0 ms.
    int recorded_max_wait_ms = -1;

    if (mode == "adaptive") {
        recorded_max_wait_ms = options.max_wait_ms;
    }
    else if (mode == "adaptive_immediate") {
        recorded_max_wait_ms = 0;
    }

    MetricsLogger metrics(
        csv_path, config, batch_size, mode, slo_config, model.name,
        recorded_max_wait_ms
    );

    if (!metrics.is_open()) {
        std::cerr << "Failed to open CSV: " << csv_path << '\n';
        return 1;
    }

    // Calibration covers every batch size from 1 up to the configured maximum,
    // because partial batches (for example at queue closure) can occur.
    std::cout << "Warm-up started.\n";
    const int warmup_runs = 3;
    const int calibration_runs = 10;
    ServiceTimeEstimator estimator(static_cast<std::size_t>(batch_size));

    for (int size = 1; size <= batch_size; ++size) {
        std::vector<Request> warmup_batch;

        for (int i = 0; i < size; ++i) {
            Request req(0);
            req.input = image_pool[i];
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

    for (std::size_t size = 1; size <= static_cast<std::size_t>(batch_size);
         ++size) {
        profile_file << size << ',' << estimator.estimate_ms(size) << ','
                     << calibration_runs << ','
                     << slo_config.safety_margin.count() << ','
                     << "prefix_max_of_calibration_max\n";
    }

    profile_file.close();
    std::cout << "Warm-up completed (batch sizes 1.." << batch_size << ").\n";
    std::cout << "Calibration saved: " << profile_path << '\n';
    std::cout << "SLO budget: " << slo_config.latency_target.count()
              << " ms | Required attainment: "
              << slo_config.target_attainment_percent
              << "% | Safety margin: "
              << slo_config.safety_margin.count() << " ms\n";
    std::cout << "Workload: " << config.name << " | "
              << config.requests_per_second << " requests/s | "
              << config.duration_seconds << " s";

    if (config.burst_on_ms > 0) {
        std::cout << " | on " << config.burst_on_ms << " ms / off "
                  << config.burst_off_ms << " ms";
    }

    std::cout << "\nImage order: request id k carries image (k-1) % "
              << image_pool.size() << '\n';

    RequestQueue request_queue;
    GeneratorStats generator_stats;

    std::cout << "MEASURE_START" << std::endl;

    std::thread producer([&]() {
        generator_stats = generate_requests(
            request_queue, image_pool, config, slo_config.latency_target
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

            // No logits are requested here: output copying is off while serving.
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
    std::cout << "Configured max batch size: " << batch_size << '\n';

    std::cout << "Scheduled requests: " << generator_stats.scheduled << '\n';
    std::cout << "Generated requests: " << generator_stats.generated << '\n';
    std::cout << "Not generated at the window boundary: "
              << generator_stats.scheduled - generator_stats.generated << '\n';
    std::cout << "Request preparation (image copy + queue push) per request: mean "
              << generator_stats.prepare_mean_us << " us, max "
              << generator_stats.prepare_max_us << " us\n";

    // Signed offset: last completion minus the end of the generation window
    // (negative when the last request finished before the window ended).
    // Drain is that offset clamped at 0: extra time spent after generation ended.
    const double last_completion_offset_ms =
        std::chrono::duration<double, std::milli>(
            metrics.last_finish_time() - generator_stats.end
        ).count();
    const double drain_ms = std::max(0.0, last_completion_offset_ms);
    const double interval_s =
        std::chrono::duration<double>(
            metrics.last_finish_time() - metrics.first_arrival_time()
        ).count();

    std::cout << "Drain after generation end: " << drain_ms << " ms\n";
    std::cout << "Last completion minus generation end (signed): "
              << last_completion_offset_ms << " ms\n";
    std::cout << "Measurement interval (first arrival to last finish): "
              << interval_s << " s (generation window "
              << config.duration_seconds << " s)\n";
    std::cout << "Peak working set: " << peak_working_set_mib() << " MiB\n";

    metrics.print_summary();

    return 0;
}

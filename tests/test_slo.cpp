#include "../include/slo_batcher.hpp"
#include "../include/static_batcher.hpp"
#include "../include/adaptive_batcher.hpp"
#include "../include/metrics.hpp"
#include "../include/request_generator.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

ServiceTimeEstimator calibrated_estimator() {
    ServiceTimeEstimator estimator(32);
    for (std::size_t size = 1; size <= 32; ++size) {
        estimator.observe(size, 1.0);
    }
    estimator.finalize();
    return estimator;
}

int main(int argc, char* argv[]) {
    const auto estimator = calibrated_estimator();
    const SloConfig config{};

    {
        ServiceTimeEstimator model(3);
        model.observe(1, 2.0);
        model.observe(1, 1.0);
        model.observe(2, 1.0);
        model.observe(3, 4.0);
        model.finalize();
        require(model.estimate_ms(1) == 2.0, "Maximum observation lost");
        require(model.estimate_ms(2) == 2.0, "Prefix envelope missing");
        require(model.estimate_ms(3) == 4.0, "Size-specific estimate lost");
        bool rejected = false;
        try {
            ServiceTimeEstimator incomplete(2);
            incomplete.observe(1, 1.0);
            incomplete.finalize();
        }
        catch (const std::logic_error&) {
            rejected = true;
        }
        require(rejected, "Missing calibration accepted");
    }

    {
        RequestQueue queue;
        Request late(1);
        late.deadline = Clock::now() - std::chrono::milliseconds(1);
        queue.push(late);
        const auto start = Clock::now();
        const auto batch = collect_slo_batch(queue, 32, estimator, config);
        require(batch.size() == 1, "Expired request was not dispatched");
        require(Clock::now() - start < std::chrono::milliseconds(100),
                "Expired request waited for a new accumulation window");
    }

    {
        RequestQueue queue;
        Request first(1);
        first.deadline = Clock::now() + std::chrono::milliseconds(40);
        queue.push(first);
        std::thread producer([&queue] {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            queue.push(Request(2, std::chrono::milliseconds(500)));
        });
        const auto batch = collect_slo_batch(queue, 32, estimator, config);
        const auto dispatched = Clock::now();
        producer.join();
        require(batch.size() == 2, "Partial batch lost requests");
        require(batch[0].id == 1 && batch[1].id == 2, "FIFO order changed");
        require(dispatched >= first.deadline - std::chrono::milliseconds(3),
                "Batch dispatched before computed slack boundary");
        require(dispatched < first.deadline + std::chrono::milliseconds(100),
                "Arrival notifications extended the oldest deadline");
    }

    {
        RequestQueue queue;
        Request first(1, std::chrono::milliseconds(500));
        queue.push(first);
        std::thread producer([&queue] {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            for (int id = 2; id <= 4; ++id) {
                queue.push(Request(id));
            }
        });
        const auto start = Clock::now();
        const auto batch = collect_slo_batch(queue, 32, estimator, config);
        const auto elapsed = Clock::now() - start;
        producer.join();
        require(batch.size() == 4, "Full target was not dispatched");
        require(elapsed < std::chrono::milliseconds(250),
                "Full target waited until the deadline");
    }

    {
        RequestQueue queue;
        queue.push(Request(1, std::chrono::milliseconds(500)));
        std::thread closer([&queue] {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            queue.close();
        });
        const auto batch = collect_slo_batch(queue, 32, estimator, config);
        closer.join();
        require(batch.size() == 1, "Closure lost partial batch");
        require(collect_slo_batch(queue, 32, estimator, config).empty(),
                "Closed empty queue did not terminate");
    }

    {
        RequestQueue queue;
        for (int id = 1; id <= 9; ++id) {
            queue.push(Request(id));
        }
        queue.close();
        const auto batch = collect_slo_batch(queue, 8, estimator, config);
        require(batch.size() == 8, "Maximum batch size was not respected");
        require(collect_slo_batch(queue, 8, estimator, config).size() == 1,
                "Closure did not drain remaining requests");
    }

    {
        RequestQueue queue;
        queue.push(Request(1));
        require(collect_slo_batch(queue, 1, estimator, config).size() == 1,
                "Batch size 1 did not dispatch immediately");
        bool rejected = false;
        try {
            collect_slo_batch(queue, 0, estimator, config);
        }
        catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "Invalid maximum batch size accepted");
    }

    {
        RequestQueue queue;
        for (int id = 1; id <= 5; ++id) {
            queue.push(Request(id));
        }
        queue.close();
        require(collect_static_batch(queue, 4).size() == 4,
                "Static baseline changed");
        require(collect_static_batch(queue, 4).size() == 1,
                "Static remainder changed");
        RequestQueue immediate;
        for (int id = 1; id <= 5; ++id) {
            immediate.push(Request(id));
        }
        immediate.close();
        require(collect_adaptive_immediate_batch(immediate, 32).size() == 4,
                "Immediate baseline changed");
        require(collect_adaptive_immediate_batch(immediate, 32).size() == 1,
                "Immediate remainder changed");
        RequestQueue waiting;
        for (int id = 1; id <= 5; ++id) {
            waiting.push(Request(id));
        }
        waiting.close();
        require(collect_adaptive_batch(waiting, 32, std::chrono::milliseconds(10)).size() == 5,
                "Waiting adaptive baseline changed");
    }

    {
        RequestQueue queue;
        const WorkloadConfig workload{"low", 2.0, 1, 0, 0};
        std::thread producer([&queue, workload] {
            generate_requests(queue, std::vector<float>{1.0f}, workload,
                              std::chrono::milliseconds(37));
        });
        int generated = 0;
        Request request(0);
        while (queue.pop(request)) {
            require(request.deadline - request.arrival_time ==
                    std::chrono::milliseconds(37),
                    "Generator did not propagate the configured budget");
            ++generated;
        }
        producer.join();
        require(generated > 0 && generated <= 2,
                "Generator did not close or emitted an invalid count");
    }

    require(argc == 2, "Pass a temporary CSV path");
    {
        MetricsLogger metrics(argv[1], WorkloadConfig{"test", 2.0, 1, 0, 0},
                              4, "slo", config);
        require(metrics.is_open(), "Could not open test CSV");
        std::vector<Request> batch{Request(1), Request(2), Request(3)};
        for (auto& request : batch) {
            request.start_service_time = request.arrival_time + std::chrono::milliseconds(18);
            request.estimated_service_ms = 3.0;
        }
        batch[0].finish_time = batch[0].deadline - std::chrono::milliseconds(1);
        batch[1].finish_time = batch[1].deadline;
        batch[2].finish_time = batch[2].deadline + std::chrono::milliseconds(1);
        metrics.record_batch(batch);
        std::ostringstream summary;
        auto* original = std::cout.rdbuf(summary.rdbuf());
        metrics.print_summary();
        std::cout.rdbuf(original);
        require(summary.str().find("SLO met requests: 2/3") != std::string::npos,
                "Deadline boundary or SLO count is wrong");
        require(summary.str().find("SLO status: FAIL") != std::string::npos,
                "Attainment threshold not applied");
    }

    std::cout << "Scheduler, baseline, estimator, and metrics checks passed.\n";
}

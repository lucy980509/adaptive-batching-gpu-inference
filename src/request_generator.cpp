#include "../include/request_generator.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thread>

GeneratorStats generate_requests(
    RequestQueue& request_queue,
    const std::vector<std::vector<float>>& image_pool,
    const WorkloadConfig& config,
    std::chrono::milliseconds latency_target
) {
    const TimePoint start = Clock::now();

    const TimePoint end =
        start + std::chrono::seconds(config.duration_seconds);

    const auto interval =
        std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(
                1.0 / config.requests_per_second
            )
        );

    if (image_pool.empty()) {
        throw std::invalid_argument("image_pool must not be empty");
    }

    GeneratorStats stats;
    stats.start = start;
    stats.end = end;
    double prepare_total_us = 0.0;

    uint64_t request_id = 1;

    auto generate_window = [&](TimePoint window_start,
                               TimePoint window_end) {
        TimePoint next_arrival = window_start;

        for (TimePoint slot = window_start; slot < window_end;
             slot += interval) {
            ++stats.scheduled;
        }

        while (next_arrival < window_end) {
            std::this_thread::sleep_until(next_arrival);

            // A slot planned inside the window is always generated, even if the
            // thread wakes up after the window ended. Dropping it would shift
            // the planned arrival time of every later request id, so the same id
            // would no longer mean the same planned arrival in every run.
            Request req(request_id++, latency_target);
            req.scheduled_arrival_time = next_arrival;

            // arrival_time was stamped in the constructor, so this copy and the
            // push below count against the request's latency and deadline.
            const TimePoint prepare_start = Clock::now();
            req.image_index = (req.id - 1) % image_pool.size();
            req.input = image_pool[req.image_index];
            request_queue.push(req);
            const double prepare_us =
                std::chrono::duration<double, std::micro>(
                    Clock::now() - prepare_start
                ).count();

            prepare_total_us += prepare_us;
            stats.prepare_max_us = std::max(stats.prepare_max_us, prepare_us);
            ++stats.generated;

            next_arrival += interval;
        }
    };

    if (config.name == "bursty") {
        const auto on_duration =
            std::chrono::milliseconds(config.burst_on_ms);

        const auto cycle_duration =
            std::chrono::milliseconds(
                config.burst_on_ms + config.burst_off_ms
            );

        for (TimePoint burst_start = start;
             burst_start < end;
             burst_start += cycle_duration) {

            const TimePoint burst_end =
                std::min(burst_start + on_duration, end);

            generate_window(burst_start, burst_end);
        }
    }
    else {
        generate_window(start, end);
    }

    std::this_thread::sleep_until(end);

    request_queue.close();

    if (stats.generated > 0) {
        stats.prepare_mean_us = prepare_total_us / stats.generated;
    }

    return stats;
}

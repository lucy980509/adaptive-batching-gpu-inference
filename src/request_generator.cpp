#include "../include/request_generator.hpp"

#include <algorithm>
#include <chrono>
#include <thread>

void generate_requests(
    RequestQueue& request_queue,
    const std::vector<float>& input_data,
    const WorkloadConfig& config
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

    uint64_t request_id = 1;

    auto generate_window = [&](TimePoint window_start,
                               TimePoint window_end) {
        TimePoint next_arrival = window_start;

        while (next_arrival < window_end) {
            std::this_thread::sleep_until(next_arrival);

            if (Clock::now() >= window_end) {
                break;
            }

            Request req(request_id++);
            req.scheduled_arrival_time = next_arrival;
            req.input = input_data;
            request_queue.push(req);

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
}
#pragma once

#include "request.hpp"
#include "service_time_estimator.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <vector>

class RequestQueue {
private:
    std::queue<Request> queue;
    std::mutex mutex;
    std::condition_variable condition;

    bool closed = false;

public:
    void push(const Request& request) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.push(request);
        }

        condition.notify_one();
    }

    bool pop(Request& request) {
        std::unique_lock<std::mutex> lock(mutex);

        condition.wait(lock, [this] {
            return !queue.empty() || closed;
        });

        if (queue.empty()) {
            return false;
        }

        request = queue.front();
        queue.pop();

        return true;
    }

    std::size_t wait_for_requests() {
        std::unique_lock<std::mutex> lock(mutex);

        condition.wait(lock, [this] {
            return !queue.empty() || closed;
        });

        return queue.size();
    }

    std::vector<Request> pop_batch(
        std::size_t target_size,
        std::chrono::milliseconds max_wait
    ) {
        std::unique_lock<std::mutex> lock(mutex);

        condition.wait(lock, [this] {
            return !queue.empty() || closed;
        });

        if (queue.empty()) {
            return {};
        }

        const TimePoint deadline =
            queue.front().arrival_time + max_wait;

        condition.wait_until(lock, deadline, [this, target_size] {
            return queue.size() >= target_size || closed;
        });

        const std::size_t count =
            std::min(target_size, queue.size());

        std::vector<Request> batch;
        batch.reserve(count);

        for (std::size_t i = 0; i < count; ++i) {
            batch.push_back(queue.front());
            queue.pop();
        }

        return batch;
    }

    std::vector<Request> pop_slo_batch(
        std::size_t max_batch_size,
        const ServiceTimeEstimator& estimator,
        std::chrono::milliseconds safety_margin
    ) {
        if (max_batch_size == 0 || max_batch_size > 32 ||
            safety_margin.count() < 0) {
            throw std::invalid_argument("Invalid SLO batch configuration");
        }

        std::unique_lock<std::mutex> lock(mutex);

        condition.wait(lock, [this] {
            return !queue.empty() || closed;
        });

        if (queue.empty()) {
            return {};
        }

        std::size_t target_size =
            std::min(std::size_t{4}, max_batch_size);

        while (target_size < queue.size() &&
               target_size < max_batch_size) {
            target_size = std::min(target_size * 2, max_batch_size);
        }

        const auto expected_service =
            std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double, std::milli>(
                    estimator.estimate_ms(target_size)
                )
            );

        const TimePoint latest_dispatch =
            queue.front().deadline - expected_service - safety_margin;

        condition.wait_until(lock, latest_dispatch, [this, target_size] {
            return queue.size() >= target_size || closed;
        });

        const std::size_t count = std::min(target_size, queue.size());
        std::vector<Request> batch;
        batch.reserve(count);

        for (std::size_t i = 0; i < count; ++i) {
            batch.push_back(queue.front());
            queue.pop();
        }

        return batch;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            closed = true;
        }

        condition.notify_all();
    }
};

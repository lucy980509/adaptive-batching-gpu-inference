#pragma once

#include "request.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
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

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            closed = true;
        }

        condition.notify_all();
    }
};
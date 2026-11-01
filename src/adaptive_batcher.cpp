#include "../include/adaptive_batcher.hpp"

#include <algorithm>
#include <stdexcept>

std::vector<Request> collect_adaptive_batch(
    RequestQueue& request_queue,
    std::size_t max_batch_size,
    std::chrono::milliseconds max_wait
) {
    if (max_batch_size == 0 || max_batch_size > 32) {
        throw std::invalid_argument(
            "max_batch_size must be between 1 and 32"
        );
    }

    if (max_wait.count() < 0) {
        throw std::invalid_argument(
            "max_wait must not be negative"
        );
    }

    const std::size_t queued_requests =
        request_queue.wait_for_requests();

    if (queued_requests == 0) {
        return {};
    }

    std::size_t target_size =
        std::min(std::size_t{4}, max_batch_size);

    while (target_size < queued_requests &&
           target_size < max_batch_size) {
        target_size = std::min(
            target_size * 2,
            max_batch_size
        );
    }

    return request_queue.pop_batch(target_size, max_wait);
}

std::vector<Request> collect_adaptive_immediate_batch(
    RequestQueue& request_queue,
    std::size_t max_batch_size
) {
    if (max_batch_size == 0 || max_batch_size > 32) {
        throw std::invalid_argument(
            "max_batch_size must be between 1 and 32"
        );
    }

    const std::size_t queued_requests =
        request_queue.wait_for_requests();

    if (queued_requests == 0) {
        return {};
    }

    const std::size_t available =
        std::min(queued_requests, max_batch_size);

    std::size_t target_size = 1;

    for (std::size_t candidate : {4, 8, 16, 32}) {
        if (candidate <= available) {
            target_size = candidate;
        }
    }

    return request_queue.pop_batch(
        target_size,
        std::chrono::milliseconds(0)
    );
}
#include "../include/static_batcher.hpp"

#include <stdexcept>

std::vector<Request> collect_static_batch(
    RequestQueue& request_queue,
    int batch_size
) {
    if (batch_size <= 0) {
        throw std::invalid_argument("batch_size must be positive");
    }

    std::vector<Request> batch;
    Request received(0);

    for (int i = 0; i < batch_size; ++i) {
        if (!request_queue.pop(received)) {
            break;
        }

        batch.push_back(received);
    }

    return batch;
}
#include "../include/slo_batcher.hpp"

#include <stdexcept>

std::vector<Request> collect_slo_batch(
    RequestQueue& request_queue,
    std::size_t max_batch_size,
    const ServiceTimeEstimator& estimator,
    const SloConfig& config
) {
    if (config.latency_target.count() <= 0 ||
        config.safety_margin.count() < 0) {
        throw std::invalid_argument("Invalid SLO timing configuration");
    }

    return request_queue.pop_slo_batch(
        max_batch_size, estimator, config.safety_margin
    );
}

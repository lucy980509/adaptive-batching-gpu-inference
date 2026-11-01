#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

class ServiceTimeEstimator {
private:
    std::vector<double> estimates_ms;

public:
    explicit ServiceTimeEstimator(std::size_t max_batch_size)
        : estimates_ms(max_batch_size + 1, 0.0) {
        if (max_batch_size == 0) {
            throw std::invalid_argument("max_batch_size must be positive");
        }
    }

    void observe(std::size_t batch_size, double service_ms) {
        if (batch_size == 0 || batch_size >= estimates_ms.size() ||
            !std::isfinite(service_ms) || service_ms <= 0.0) {
            throw std::invalid_argument("Invalid service time observation");
        }

        estimates_ms[batch_size] = std::max(
            estimates_ms[batch_size], service_ms
        );
    }

    void finalize() {
        for (std::size_t size = 1; size < estimates_ms.size(); ++size) {
            if (estimates_ms[size] <= 0.0) {
                throw std::logic_error("Missing batch-size calibration");
            }

            estimates_ms[size] = std::max(
                estimates_ms[size], estimates_ms[size - 1]
            );
        }
    }

    double estimate_ms(std::size_t batch_size) const {
        if (batch_size == 0 || batch_size >= estimates_ms.size() ||
            estimates_ms[batch_size] <= 0.0) {
            throw std::out_of_range("Batch size is not calibrated");
        }

        return estimates_ms[batch_size];
    }
};

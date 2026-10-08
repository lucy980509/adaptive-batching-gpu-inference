#include <chrono>
#include <cstdint>
#include <vector>

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

struct Request {
    uint64_t id;
    TimePoint arrival_time;

    std::vector<float> input;
    TimePoint start_service_time;
    TimePoint finish_time;

    Request(uint64_t request_id){
        id = request_id;
        arrival_time = Clock::now();
    }
};
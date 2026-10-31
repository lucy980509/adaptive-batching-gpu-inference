#pragma once

#include "request.hpp"

#include <condition_variable>
#include <mutex>
#include <queue>

class RequestQueue {
private:
    std::queue<Request> queue;
    std::mutex mutex;
    std::condition_variable condition;

public:
    void push(const Request& request) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.push(request);
        }

        condition.notify_one();
    }

        Request pop() {
        std::unique_lock<std::mutex> lock(mutex);

        condition.wait(lock, [this] {
            return !queue.empty();
        });

        Request request = queue.front();
        queue.pop();

        return request;
    }
};
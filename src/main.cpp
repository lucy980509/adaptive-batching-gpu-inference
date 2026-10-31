#include "../include/request_queue.hpp"

#include <chrono>
#include <iostream>
#include <thread>

int main() {
    RequestQueue request_queue;
    const int request_count = 5;

    std::thread producer([&request_queue, request_count]() {
        for (int i = 1; i <= request_count; ++i) {
            Request req(i);
            request_queue.push(req);

            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });

    for (int i = 0; i < request_count; ++i) {
        Request received = request_queue.pop();
        std::cout << "Received request ID: " << received.id << '\n';
    }

    producer.join();
    return 0;
}
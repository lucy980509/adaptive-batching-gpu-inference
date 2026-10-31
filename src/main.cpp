#include "../include/request_queue.hpp"
#include <iostream>

int main() {
    Request req(1);
    RequestQueue request_Queue;
    request_Queue.push(req);
    Request received = request_Queue.pop();
    
    TimePoint now = Clock::now();

    auto duration = now - req.arrival_time; 
    
    auto elapsed =std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    
    std::cout << "received request id: " << received.id << '\n';
    std::cout << elapsed.count() << " ms\n";
    
    return 0;

}
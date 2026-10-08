#include "../include/request.hpp"
#include <iostream>

int main() {
    Request req(1);

    TimePoint now = Clock::now();

    auto duration = now - req.arrival_time; 
    
    auto elapsed =std::chrono::duration_cast<std::chrono::milliseconds>(duration);

    std::cout << elapsed.count() << " ms\n";
    return 0;
}
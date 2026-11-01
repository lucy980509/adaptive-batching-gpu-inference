#include "../include/request_queue.hpp"
#include "../include/preprocess.hpp"
#include "../include/inference_engine.hpp"

#include <chrono>
#include <thread>
#include <vector>

int main() {
    InferenceEngine engine(L"models/wafer_fault_cnn.onnx");

    std::vector<float> input_data =
        preprocess_image("test_images/sample.png");

    if (input_data.empty()) {
        return 1;
    }

    RequestQueue request_queue;
    const int request_count = 32;
    const int batch_size = 16;

    std::thread producer([&request_queue, request_count, &input_data]() {
        for (int i = 1; i <= request_count; ++i) {
            Request req(i);
            req.input = input_data;
            request_queue.push(req);

            std::this_thread::sleep_for(
                std::chrono::milliseconds(500)
            );
        }
    });

    std::vector<Request> batch;

    for (int i = 0; i < request_count; ++i) {
        Request received = request_queue.pop();
        batch.push_back(received);

        if (batch.size() == batch_size || i == request_count - 1) {
            engine.run_batch(batch);
            batch.clear();
        }
    }

    producer.join();
    return 0;
}
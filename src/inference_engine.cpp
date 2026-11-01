#include "../include/inference_engine.hpp"

#include <iostream>

InferenceEngine::InferenceEngine(const std::wstring& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "P3Inference") {

    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;

    session_options.AppendExecutionProvider_CUDA(cuda_options);

    std::cout << "CUDA Execution Provider configured.\n";

    session = Ort::Session(
        env,
        model_path.c_str(),
        session_options
    );

    std::cout << "ONNX model loaded successfully.\n";

    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name_alloc =
        session.GetInputNameAllocated(0, allocator);

    auto output_name_alloc =
        session.GetOutputNameAllocated(0, allocator);

    input_name = input_name_alloc.get();
    output_name = output_name_alloc.get();
}

void InferenceEngine::run_batch(const std::vector<Request>& batch) {
    if (batch.empty()) {
        return;
    }

    std::vector<float> batch_input;

    for (const Request& request : batch) {
        batch_input.insert(
            batch_input.end(),
            request.input.begin(),
            request.input.end()
        );
    }

    std::cout << "Batch input values: "
              << batch_input.size() << '\n';

    std::vector<int64_t> input_shape = {
        static_cast<int64_t>(batch.size()),
        1,
        24,
        24
    };

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault
    );

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        batch_input.data(),
        batch_input.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = { input_name.c_str() };
    const char* output_names[] = { output_name.c_str() };

    std::cout << "Batch:";

    for (const Request& request : batch) {
        std::cout << ' ' << request.id;
    }

    std::cout << '\n';

    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    std::cout << "Inference completed for "
              << batch.size() << " requests.\n";
}
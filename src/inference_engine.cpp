#include "../include/inference_engine.hpp"

#include <iostream>
#include <stdexcept>

InferenceEngine::InferenceEngine(const ModelConfig& model_config)
    : env(ORT_LOGGING_LEVEL_WARNING, "P3Inference"),
      model(model_config) {

    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;

    session_options.AppendExecutionProvider_CUDA(cuda_options);

    std::cout << "CUDA Execution Provider configured.\n";

    session = Ort::Session(
        env,
        model.model_path.c_str(),
        session_options
    );

    std::cout << "ONNX model loaded successfully: " << model.name << '\n';

    // The model must report the input the configuration promises.
    if (session.GetInputCount() != 1) {
        throw std::runtime_error(
            "Model '" + model.name + "' must have exactly one input, but has " +
            std::to_string(session.GetInputCount())
        );
    }

    const auto input_info =
        session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> reported_shape = input_info.GetShape();

    std::string reported_text = "[";
    for (std::size_t i = 0; i < reported_shape.size(); ++i) {
        reported_text += (i == 0 ? "" : ",") +
                         std::to_string(reported_shape[i]);
    }
    reported_text += "]";

    const std::string expected_text =
        "[N," + std::to_string(model.channels) + "," +
        std::to_string(model.height) + "," +
        std::to_string(model.width) + "]";

    const bool shape_matches =
        reported_shape.size() == 4 &&
        reported_shape[0] == -1 &&
        reported_shape[1] == model.channels &&
        reported_shape[2] == model.height &&
        reported_shape[3] == model.width;

    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error(
            "Model '" + model.name + "' input must be float32"
        );
    }

    if (!shape_matches) {
        throw std::runtime_error(
            "Model '" + model.name + "' input shape mismatch: model reports " +
            reported_text + " (-1 = dynamic), configuration expects " +
            expected_text + " with a dynamic batch axis"
        );
    }

    std::cout << "Input shape verified: " << reported_text << " float32\n";

    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name_alloc =
        session.GetInputNameAllocated(0, allocator);

    auto output_name_alloc =
        session.GetOutputNameAllocated(0, allocator);

    input_name = input_name_alloc.get();
    output_name = output_name_alloc.get();
}

void InferenceEngine::run_batch(
    std::vector<Request>& batch,
    std::vector<float>* output_values
) {
    if (batch.empty()) {
        return;
    }

    for (const Request& request : batch) {
        if (request.input.size() != model.input_elements()) {
            throw std::invalid_argument(
                "Request " + std::to_string(request.id) + " has " +
                std::to_string(request.input.size()) +
                " input values, but model '" + model.name + "' expects " +
                std::to_string(model.input_elements())
            );
        }
    }

    TimePoint start_time = Clock::now();

    for (Request& request : batch) {
        request.start_service_time = start_time;
    }

    std::vector<float> batch_input;

    for (const Request& request : batch) {
        batch_input.insert(
            batch_input.end(),
            request.input.begin(),
            request.input.end()
        );
    }

    std::vector<int64_t> input_shape = {
        static_cast<int64_t>(batch.size()),
        model.channels,
        model.height,
        model.width
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

    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    TimePoint finish_time = Clock::now();

    for (Request& request : batch) {
        request.finish_time = finish_time;
    }

    // Copied after finish_time so the optional copy is not counted as service time.
    if (output_values != nullptr) {
        const float* data = outputs[0].GetTensorData<float>();
        const std::size_t count =
            outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        output_values->assign(data, data + count);
    }
}
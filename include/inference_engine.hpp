#pragma once

#include "model_config.hpp"
#include "request.hpp"

#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

class InferenceEngine {
private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session{nullptr};

    ModelConfig model;
    std::string input_name;
    std::string output_name;

public:
    InferenceEngine(const ModelConfig& model_config);

    // If output_values is not null, the first output tensor is copied into it.
    void run_batch(
        std::vector<Request>& batch,
        std::vector<float>* output_values = nullptr
    );
};

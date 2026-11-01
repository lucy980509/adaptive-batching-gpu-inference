#pragma once

#include "request.hpp"

#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

class InferenceEngine {
private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session{nullptr};

    std::string input_name;
    std::string output_name;

public:
    InferenceEngine(const std::wstring& model_path);

    void run_batch(std::vector<Request>& batch);
};
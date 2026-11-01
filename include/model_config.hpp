#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

// Everything the serving code needs to know about one model.
// The ONNX input is expected to be float32 with shape [N, channels, height, width].
struct ModelConfig {
    std::string name;
    std::wstring model_path;
    int64_t channels;
    int64_t height;
    int64_t width;

    std::size_t input_elements() const {
        return static_cast<std::size_t>(channels * height * width);
    }
};

inline ModelConfig get_model_config(const std::string& name) {
    if (name == "wafer") {
        return {"wafer", L"models/wafer_fault_cnn.onnx", 1, 24, 24};
    }

    if (name == "resnet50") {
        return {"resnet50", L"models/resnet50/resnet50_v2_fp32.onnx", 3, 224, 224};
    }

    throw std::invalid_argument("Unknown model: " + name);
}

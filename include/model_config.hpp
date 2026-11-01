#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// Input specification of the served model. The ONNX input is expected to be
// float32 with shape [N, channels, height, width] and a dynamic batch axis.
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

// ResNet-50 (torchvision IMAGENET1K_V2) exported by scripts/prepare_resnet50.py.
inline ModelConfig resnet50_model_config() {
    return {"resnet50", L"models/resnet50/resnet50_v2_fp32.onnx", 3, 224, 224};
}

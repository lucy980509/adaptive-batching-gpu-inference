#pragma once

#include <cstddef>
#include <string>
#include <vector>

// The prepared ImageNet subset written by scripts/prepare_imagenet_subset.py:
// raw float32 little-endian, [image_count, floats_per_image], no header.
const std::string kPreparedImagePath =
    "data/imagenet/val1000_resnet50_v2_nchw_f32.bin";
const std::size_t kPreparedImageCount = 1000;

// Reads `count` consecutive images starting at image `first`, without loading
// the rest of the file. Throws std::runtime_error with a clear message if the
// file is missing, has the wrong size, or the range is outside the file.
std::vector<std::vector<float>> read_prepared_images(
    const std::string& path,
    std::size_t image_count,
    std::size_t floats_per_image,
    std::size_t first,
    std::size_t count
);

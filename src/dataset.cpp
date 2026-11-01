#include "../include/dataset.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>

std::vector<std::vector<float>> read_prepared_images(
    const std::string& path,
    std::size_t image_count,
    std::size_t floats_per_image,
    std::size_t first,
    std::size_t count
) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(
            "Prepared image file not found: " + path +
            " (run scripts/prepare_imagenet_subset.py)"
        );
    }

    const std::uintmax_t expected_bytes =
        static_cast<std::uintmax_t>(image_count) * floats_per_image *
        sizeof(float);
    const std::uintmax_t actual_bytes = std::filesystem::file_size(path);

    if (actual_bytes != expected_bytes) {
        throw std::runtime_error(
            "Prepared image file " + path + " has " +
            std::to_string(actual_bytes) + " bytes, expected " +
            std::to_string(expected_bytes) + " (" +
            std::to_string(image_count) + " images x " +
            std::to_string(floats_per_image) + " float32)"
        );
    }

    if (count == 0 || first + count > image_count) {
        throw std::runtime_error(
            "Requested images [" + std::to_string(first) + ", " +
            std::to_string(first + count) + ") are outside 0.." +
            std::to_string(image_count)
        );
    }

    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open " + path);
    }

    // The file is little-endian float32, which is also this x64 machine's layout.
    file.seekg(
        static_cast<std::streamoff>(first * floats_per_image * sizeof(float))
    );

    std::vector<std::vector<float>> images(
        count, std::vector<float>(floats_per_image)
    );

    for (std::vector<float>& image : images) {
        file.read(
            reinterpret_cast<char*>(image.data()),
            static_cast<std::streamsize>(floats_per_image * sizeof(float))
        );

        if (!file) {
            throw std::runtime_error("Failed to read images from " + path);
        }
    }

    return images;
}

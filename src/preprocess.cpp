#include "../include/preprocess.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>

std::vector<float> preprocess_image(const std::string& image_path) {
    cv::Mat image = cv::imread(
        image_path,
        cv::IMREAD_GRAYSCALE
    );

    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << '\n';
        return {};
    }

    std::cout << "Image loaded: "
              << image.cols << " x " << image.rows << '\n';

    cv::Mat image_float;
    image.convertTo(image_float, CV_32F);

    double max_value;
    cv::minMaxLoc(image_float, nullptr, &max_value);

    if (max_value > 0.0) {
        image_float /= static_cast<float>(max_value);
    }

    cv::Mat image_resized;
    cv::resize(
        image_float,
        image_resized,
        cv::Size(24, 24),
        0,
        0,
        cv::INTER_LINEAR
    );

    std::vector<float> input_data(
        image_resized.ptr<float>(),
        image_resized.ptr<float>() + image_resized.total()
    );

    return input_data;
}
#include "image_preprocess.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include <opencv2/imgproc.hpp>

namespace edge_ai {

PreprocessedInput preprocessImage(const cv::Mat& bgr, int imgsz)
{
    if (bgr.empty()) {
        throw std::runtime_error("input image is empty");
    }
    if (bgr.type() != CV_8UC3) {
        throw std::runtime_error(
            "input image must be CV_8UC3 (BGR); got OpenCV type " +
            std::to_string(bgr.type()));
    }
    if (imgsz <= 0) {
        throw std::runtime_error("input size must be positive");
    }

    const int original_width = bgr.cols;
    const int original_height = bgr.rows;

    // Use double precision for ratio/new dimensions to match the Python
    // reference arithmetic; only the final tensor values are float32.
    const double ratio = static_cast<double>(imgsz) /
                         static_cast<double>(std::max(original_width, original_height));
    const int new_width = static_cast<int>(original_width * ratio);
    const int new_height = static_cast<int>(original_height * ratio);
    if (new_width <= 0 || new_height <= 0) {
        throw std::runtime_error("letterbox resize produced an empty image");
    }

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(new_width, new_height), 0.0, 0.0,
               cv::INTER_LINEAR);

    const int pad_width = imgsz - new_width;
    const int pad_height = imgsz - new_height;
    const int left = pad_width / 2;
    const int top = pad_height / 2;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, pad_height - top, left,
                       pad_width - left, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));

    cv::Mat float_image;
    padded.convertTo(float_image, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels;
    cv::split(float_image, channels);
    if (channels.size() != 3) {
        throw std::runtime_error("expected 3 BGR channels after preprocessing");
    }

    const std::size_t plane = static_cast<std::size_t>(imgsz) *
                              static_cast<std::size_t>(imgsz);
    PreprocessedInput result;
    result.data.resize(3 * plane);
    for (int c = 0; c < 3; ++c) {
        cv::Mat channel = channels[static_cast<std::size_t>(c)];
        if (!channel.isContinuous()) {
            channel = channel.clone();
        }
        std::memcpy(result.data.data() + static_cast<std::size_t>(c) * plane,
                    channel.ptr<float>(), plane * sizeof(float));
    }
    result.letterbox = LetterboxInfo{ratio, left, top, original_width,
                                     original_height};
    return result;
}

}  // namespace edge_ai

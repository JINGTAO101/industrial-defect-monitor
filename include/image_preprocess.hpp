#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "model_contract.hpp"

namespace edge_ai {

struct LetterboxInfo {
    double ratio = 1.0;
    int pad_left = 0;
    int pad_top = 0;
    int original_width = 0;
    int original_height = 0;
};

struct PreprocessedInput {
    std::vector<float> data;  // NCHW, 1 x 3 x imgsz x imgsz
    LetterboxInfo letterbox;
};

// Matches the verified 2A reference:
// BGR 8UC3 -> letterbox (INTER_LINEAR, pad 114) -> float32 / 255 -> CHW -> NCHW.
// Throws std::runtime_error for empty/non-8UC3 input or invalid sizes.
PreprocessedInput preprocessImage(const cv::Mat& bgr, int imgsz = kInputSize);

}  // namespace edge_ai

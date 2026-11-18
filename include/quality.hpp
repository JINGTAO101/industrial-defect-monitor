#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "postprocess.hpp"

namespace edge_ai {

struct QualityResult {
    bool is_ng = false;
    float confidence_threshold = 0.0f;
    std::size_t total_detections = 0;
    std::vector<Detection> defects;  // detections with confidence > threshold
};

// Applies the SRS rule: NG if at least one valid detection has confidence
// strictly greater than the threshold; otherwise OK. Invalid thresholds
// (non-finite, <= 0, > 1) return false with a diagnostic and no decision.
bool evaluateQuality(
    const std::vector<Detection>& detections,
    float confidence_threshold,
    QualityResult& result,
    std::string& error);

const char* qualityLabel(bool is_ng);

}  // namespace edge_ai

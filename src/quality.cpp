#include "quality.hpp"

#include <cmath>

namespace edge_ai {

bool evaluateQuality(
    const std::vector<Detection>& detections,
    float confidence_threshold,
    QualityResult& result,
    std::string& error)
{
    error.clear();
    result = QualityResult{};
    if (!std::isfinite(confidence_threshold) ||
        !(confidence_threshold > 0.0f && confidence_threshold <= 1.0f)) {
        error = "confidence threshold must be finite and in (0, 1]";
        return false;
    }

    result.confidence_threshold = confidence_threshold;
    result.total_detections = detections.size();
    for (const Detection& detection : detections) {
        if (detection.confidence > confidence_threshold) {
            result.defects.push_back(detection);
        }
    }
    result.is_ng = !result.defects.empty();
    return true;
}

const char* qualityLabel(bool is_ng)
{
    return is_ng ? "NG" : "OK";
}

}  // namespace edge_ai

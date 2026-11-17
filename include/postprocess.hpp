#pragma once

#include <map>
#include <string>
#include <vector>

#include "image_preprocess.hpp"

namespace edge_ai {

struct Detection {
    int class_id = -1;
    std::string class_name;
    float confidence = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

struct PostprocessOptions {
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
};

struct PostprocessResult {
    std::vector<Detection> detections;
    std::vector<std::string> anomalies;
    std::string error;  // empty on success
};

// Decodes [1, 4+nc, 8400] channel-first output. Returns an error for
// non-finite values, size/class-count mismatch, or invalid letterbox data.
PostprocessResult postprocess(
    const std::vector<float>& output,
    const std::map<int, std::string>& class_names,
    const LetterboxInfo& letterbox,
    const PostprocessOptions& options);

// FR-007: unknown ids map to "Unknown" and append an anomaly; never silently
// map to an existing class.
std::string classNameFor(
    int class_id,
    const std::map<int, std::string>& class_names,
    std::vector<std::string>& anomalies);

// Class-aware greedy NMS (the same order as the 2A reference: classes in
// ascending order, descending score within each class).
std::vector<Detection> classAwareNms(
    const std::vector<Detection>& detections,
    float iou_threshold);

}  // namespace edge_ai

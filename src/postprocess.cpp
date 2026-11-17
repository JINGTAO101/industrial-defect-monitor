#include "postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>

#include <opencv2/dnn.hpp>

namespace edge_ai {

std::string classNameFor(
    int class_id,
    const std::map<int, std::string>& class_names,
    std::vector<std::string>& anomalies)
{
    const auto it = class_names.find(class_id);
    if (it != class_names.end()) {
        return it->second;
    }
    std::ostringstream message;
    message << "unknown class id " << class_id
            << "; displaying 'Unknown' and recording an anomaly";
    anomalies.push_back(message.str());
    return "Unknown";
}

std::vector<Detection> classAwareNms(
    const std::vector<Detection>& detections,
    float iou_threshold)
{
    std::set<int> class_ids;
    for (const auto& detection : detections) {
        class_ids.insert(detection.class_id);
    }

    std::vector<Detection> kept;
    for (const int class_id : class_ids) {
        std::vector<cv::Rect2d> boxes;
        std::vector<float> scores;
        std::vector<std::size_t> original_indices;
        for (std::size_t i = 0; i < detections.size(); ++i) {
            const Detection& detection = detections[i];
            if (detection.class_id != class_id) {
                continue;
            }
            boxes.emplace_back(
                static_cast<double>(detection.x1),
                static_cast<double>(detection.y1),
                static_cast<double>(detection.x2 - detection.x1),
                static_cast<double>(detection.y2 - detection.y1));
            scores.push_back(detection.confidence);
            original_indices.push_back(i);
        }

        std::vector<int> indices;
        // Candidates were already filtered with a strict "> conf" test.
        cv::dnn::NMSBoxes(boxes, scores, 0.0f, iou_threshold, indices);
        for (const int index : indices) {
            kept.push_back(detections[original_indices[static_cast<std::size_t>(index)]]);
        }
    }
    return kept;
}

PostprocessResult postprocess(
    const std::vector<float>& output,
    const std::map<int, std::string>& class_names,
    const LetterboxInfo& letterbox,
    const PostprocessOptions& options)
{
    PostprocessResult result;

    if (class_names.empty()) {
        result.error = "class mapping is empty";
        return result;
    }
    if (!(letterbox.ratio > 0.0)) {
        result.error = "letterbox ratio must be positive";
        return result;
    }

    const std::size_t channels = 4 + class_names.size();
    const std::size_t expected_size = channels * kNumPredictions;
    if (output.size() != expected_size) {
        std::ostringstream message;
        message << "output tensor size mismatch: expected " << expected_size
                << " floats (" << channels << " channels x " << kNumPredictions
                << " predictions), got " << output.size();
        result.error = message.str();
        return result;
    }

    for (std::size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(output[i])) {
            result.error = "non-finite value in output tensor at index " +
                           std::to_string(i);
            return result;
        }
    }

    // The verified model emits Sigmoid class probabilities; values outside
    // [0, 1] indicate a broken/incompatible model and must not become
    // detections.
    for (std::size_t i = 0; i < static_cast<std::size_t>(kNumPredictions); ++i) {
        for (std::size_t c = 0; c < class_names.size(); ++c) {
            const float score = output[(4 + c) * kNumPredictions + i];
            if (score < 0.0f || score > 1.0f) {
                std::ostringstream message;
                message << "class score out of range [0, 1] at prediction "
                        << i << ", class " << c << ": " << score;
                result.error = message.str();
                return result;
            }
        }
    }

    if (!(options.conf_threshold > 0.0f && options.conf_threshold <= 1.0f)) {
        result.error = "confidence threshold must be in (0, 1]";
        return result;
    }
    if (!(options.iou_threshold > 0.0f && options.iou_threshold <= 1.0f)) {
        result.error = "IoU threshold must be in (0, 1]";
        return result;
    }

    const float ratio = static_cast<float>(letterbox.ratio);
    const float max_x = static_cast<float>(letterbox.original_width);
    const float max_y = static_cast<float>(letterbox.original_height);
    const float pad_left = static_cast<float>(letterbox.pad_left);
    const float pad_top = static_cast<float>(letterbox.pad_top);

    std::vector<Detection> candidates;
    candidates.reserve(32);
    for (std::size_t i = 0; i < static_cast<std::size_t>(kNumPredictions); ++i) {
        const float cx = output[0 * kNumPredictions + i];
        const float cy = output[1 * kNumPredictions + i];
        const float width = output[2 * kNumPredictions + i];
        const float height = output[3 * kNumPredictions + i];

        float best_score = -std::numeric_limits<float>::infinity();
        int best_class = -1;
        for (std::size_t c = 0; c < class_names.size(); ++c) {
            const float score = output[(4 + c) * kNumPredictions + i];
            if (score > best_score) {
                best_score = score;
                best_class = static_cast<int>(c);
            }
        }
        if (best_class < 0 || !(best_score > options.conf_threshold)) {
            continue;
        }

        float x1 = (cx - width / 2.0f - pad_left) / ratio;
        float y1 = (cy - height / 2.0f - pad_top) / ratio;
        float x2 = (cx + width / 2.0f - pad_left) / ratio;
        float y2 = (cy + height / 2.0f - pad_top) / ratio;

        x1 = std::clamp(x1, 0.0f, max_x);
        y1 = std::clamp(y1, 0.0f, max_y);
        x2 = std::clamp(x2, 0.0f, max_x);
        y2 = std::clamp(y2, 0.0f, max_y);

        if (x2 <= x1 || y2 <= y1) {
            std::ostringstream message;
            message << "skipped degenerate box at prediction " << i
                    << " after coordinate restoration";
            result.anomalies.push_back(message.str());
            continue;
        }

        Detection detection;
        detection.class_id = best_class;
        detection.class_name = classNameFor(best_class, class_names, result.anomalies);
        detection.confidence = best_score;
        detection.x1 = x1;
        detection.y1 = y1;
        detection.x2 = x2;
        detection.y2 = y2;
        candidates.push_back(std::move(detection));
    }

    try {
        result.detections = classAwareNms(candidates, options.iou_threshold);
    } catch (const cv::Exception& exception) {
        result.error = std::string("NMS failed: ") + exception.what();
        result.detections.clear();
    }
    return result;
}

}  // namespace edge_ai

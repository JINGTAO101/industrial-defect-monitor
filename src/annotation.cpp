#include "annotation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace edge_ai {
namespace {

constexpr int kStatusPadding = 6;
constexpr int kStatusMargin = 8;

cv::Scalar statusColor(const AnnotationInput& input,
                       const AnnotationStyle& style)
{
    if (input.has_error) {
        return style.error_color;
    }
    if (input.quality_available && input.is_ng) {
        return style.ng_color;
    }
    return style.ok_color;
}

std::string statusText(const AnnotationInput& input)
{
    if (input.has_error) {
        return "ERROR";
    }
    if (!input.quality_available) {
        return "N/A";
    }
    return input.is_ng ? "NG" : "OK";
}

std::string confidenceLabel(const Detection& detection)
{
    char buffer[64] = {};
    std::snprintf(buffer, sizeof(buffer), "%.2f",
                  static_cast<double>(detection.confidence));
    return detection.class_name + " " + buffer;
}

// Clamp and empty-box checks keep malformed detections from drawing outside
// the image or inverting the rectangle. Detections are already in original
// image coordinates; this function deliberately applies no scaling.
bool normalizedBox(const Detection& detection,
                   int width,
                   int height,
                   cv::Rect& box)
{
    if (!std::isfinite(detection.x1) || !std::isfinite(detection.y1) ||
        !std::isfinite(detection.x2) || !std::isfinite(detection.y2)) {
        return false;
    }
    const int x1 = std::clamp(static_cast<int>(std::lround(detection.x1)), 0,
                              std::max(0, width - 1));
    const int y1 = std::clamp(static_cast<int>(std::lround(detection.y1)), 0,
                              std::max(0, height - 1));
    const int x2 = std::clamp(static_cast<int>(std::lround(detection.x2)), 0,
                              std::max(0, width - 1));
    const int y2 = std::clamp(static_cast<int>(std::lround(detection.y2)), 0,
                              std::max(0, height - 1));
    if (x2 <= x1 || y2 <= y1) {
        return false;
    }
    box = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
    return true;
}

constexpr int kLabelPadding = 3;

struct LabelGeometry {
    cv::Size text_size;
    int baseline = 0;
    int width = 0;
    int height = 0;
};

LabelGeometry labelGeometry(const std::string& text,
                            const AnnotationStyle& style)
{
    LabelGeometry geometry;
    geometry.text_size = cv::getTextSize(
        text, cv::FONT_HERSHEY_SIMPLEX, style.font_scale,
        style.font_thickness, &geometry.baseline);
    geometry.width = geometry.text_size.width + 2 * kLabelPadding;
    geometry.height = geometry.text_size.height + geometry.baseline +
                      2 * kLabelPadding;
    return geometry;
}

cv::Rect clampedLabel(const cv::Rect& label, int image_width)
{
    cv::Rect result = label;
    result.width = std::min(result.width, image_width);
    result.x = std::clamp(result.x, 0, std::max(0, image_width - result.width));
    return result;
}

bool overlaps(const cv::Rect& first, const cv::Rect& second)
{
    return (first & second).area() > 0;
}

cv::Rect statusRect(const cv::Size& size,
                    const AnnotationInput& input,
                    const AnnotationStyle& style)
{
    const std::string status = statusText(input);
    int baseline = 0;
    const cv::Size text_size =
        cv::getTextSize(status, cv::FONT_HERSHEY_SIMPLEX,
                        style.font_scale * 1.2, style.font_thickness + 1,
                        &baseline);
    const int width =
        std::min(size.width, text_size.width + 2 * kStatusMargin);
    const int height =
        std::min(size.height, text_size.height + baseline +
                                  2 * kStatusPadding);
    return cv::Rect(0, 0, std::max(0, width), std::max(0, height));
}

void drawLabelAt(cv::Mat& image,
                 const std::string& text,
                 const cv::Rect& label,
                 const cv::Scalar& color,
                 const AnnotationStyle& style)
{
    if (label.width <= 0 || label.height <= 0) {
        return;
    }
    const LabelGeometry geometry = labelGeometry(text, style);
    cv::rectangle(image, label, color, cv::FILLED);
    cv::putText(image, text,
                cv::Point(label.x + kLabelPadding,
                          label.y + kLabelPadding + geometry.text_size.height),
                cv::FONT_HERSHEY_SIMPLEX, style.font_scale,
                cv::Scalar(255, 255, 255), style.font_thickness, cv::LINE_8);
}

}  // namespace

AnnotationLayout computeAnnotationLayout(
    const cv::Size& image_size,
    const AnnotationInput& input,
    const AnnotationStyle& style)
{
    AnnotationLayout layout;
    layout.status = statusRect(image_size, input, style);
    if (input.has_error || !input.quality_available) {
        return layout;
    }

    for (const Detection& detection : input.detections) {
        cv::Rect box;
        if (!normalizedBox(detection, image_size.width, image_size.height,
                           box)) {
            continue;
        }
        const std::string text = confidenceLabel(detection);
        const LabelGeometry geometry = labelGeometry(text, style);

        const auto candidate = [&](int x, int top) {
            return clampedLabel(
                cv::Rect(x, top, geometry.width, geometry.height),
                image_size.width);
        };

        std::vector<cv::Rect> placements;
        placements.push_back(candidate(box.x, box.y - geometry.height - 2));
        int inside_top = box.y + 2;
        if (overlaps(candidate(box.x, inside_top), layout.status)) {
            inside_top = layout.status.y + layout.status.height + 2;
        }
        placements.push_back(candidate(box.x, inside_top));
        placements.push_back(
            candidate(box.x, box.y + box.height + 2));
        placements.push_back(
            candidate(box.x, box.y + box.height - geometry.height - 2));
        placements.push_back(
            candidate(box.x + box.width + 3, box.y));
        placements.push_back(
            candidate(box.x - geometry.width - 3, box.y));
        placements.push_back(
            candidate(box.x, layout.status.y + layout.status.height + 2));

        const auto conflicts = [&](const cv::Rect& placement) {
            if (overlaps(placement, layout.status)) {
                return true;
            }
            for (const cv::Rect& placed : layout.labels) {
                if (overlaps(placement, placed)) {
                    return true;
                }
            }
            return false;
        };

        cv::Rect chosen;
        bool found = false;
        for (const cv::Rect& placement : placements) {
            if (placement.y >= 0 &&
                placement.y + placement.height <= image_size.height &&
                !conflicts(placement)) {
                chosen = placement;
                found = true;
                break;
            }
        }
        if (!found) {
            // Scan downward for a free slot when every preferred position is
            // taken by the banner or an earlier label.
            const int start = std::max(
                0, layout.status.y + layout.status.height + 2);
            for (int top = start;
                 top + geometry.height <= image_size.height; top += 4) {
                const cv::Rect candidate_rect = clampedLabel(
                    cv::Rect(box.x, top, geometry.width, geometry.height),
                    image_size.width);
                if (!conflicts(candidate_rect)) {
                    chosen = candidate_rect;
                    found = true;
                    break;
                }
            }
        }
        if (!found) {
            // Dense images: keep the label inside the image even if it must
            // overlap; this is a last resort.
            const int top = std::min(
                std::max(0, layout.status.y + layout.status.height + 2),
                std::max(0, image_size.height - geometry.height));
            chosen = clampedLabel(
                cv::Rect(box.x, top, geometry.width,
                         std::min(geometry.height, image_size.height)),
                image_size.width);
        }
        layout.labels.push_back(chosen);
    }
    return layout;
}

cv::Mat annotateFrame(const cv::Mat& image,
                      const AnnotationInput& input,
                      const AnnotationStyle& style)
{
    if (image.empty()) {
        return {};
    }

    cv::Mat annotated;
    if (image.type() == CV_8UC3) {
        annotated = image.clone();
    } else if (image.type() == CV_8UC1) {
        cv::cvtColor(image, annotated, cv::COLOR_GRAY2BGR);
    } else if (image.type() == CV_8UC4) {
        cv::cvtColor(image, annotated, cv::COLOR_BGRA2BGR);
    } else {
        return {};
    }

    const cv::Scalar color = statusColor(input, style);
    const AnnotationLayout layout =
        computeAnnotationLayout(annotated.size(), input, style);

    // Detections are only drawn for completed quality decisions. A failed
    // frame shows ERROR only and must never imply an OK/NG result.
    if (!input.has_error && input.quality_available) {
        std::size_t label_index = 0;
        for (const Detection& detection : input.detections) {
            cv::Rect box;
            if (!normalizedBox(detection, annotated.cols, annotated.rows,
                               box)) {
                continue;
            }
            cv::rectangle(annotated, box, color, style.box_thickness,
                          cv::LINE_8);
            if (label_index < layout.labels.size()) {
                drawLabelAt(annotated, confidenceLabel(detection),
                            layout.labels[label_index++], color, style);
            }
        }
    }

    // Draw the banner after labels; placement already guarantees no overlap.
    if (layout.status.width > 0 && layout.status.height > 0) {
        cv::rectangle(annotated, layout.status, color, cv::FILLED);
        cv::putText(annotated, statusText(input),
                    cv::Point(kStatusMargin,
                              layout.status.height - kStatusPadding),
                    cv::FONT_HERSHEY_SIMPLEX, style.font_scale * 1.2,
                    cv::Scalar(255, 255, 255), style.font_thickness + 1,
                    cv::LINE_8);
    }

    return annotated;
}

bool writeAnnotatedImage(const cv::Mat& annotated,
                         const std::string& path,
                         std::string& error)
{
    if (annotated.empty()) {
        error = "annotated image is empty: " + path;
        return false;
    }
    const std::filesystem::path parent =
        std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::error_code fs_error;
        std::filesystem::create_directories(parent, fs_error);
        if (fs_error) {
            error = "cannot create directory '" + parent.string() + "': " +
                    fs_error.message();
            return false;
        }
    }
    try {
        if (!cv::imwrite(path, annotated)) {
            error = "failed to write annotated image: " + path;
            return false;
        }
    } catch (const cv::Exception& exception) {
        error = "failed to write annotated image: " + path + ": " +
                exception.what();
        return false;
    }
    return true;
}

}  // namespace edge_ai

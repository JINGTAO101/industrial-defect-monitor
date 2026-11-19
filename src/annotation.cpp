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

void drawLabel(cv::Mat& image,
               const std::string& text,
               int x,
               int baseline_y,
               const cv::Scalar& color,
               const AnnotationStyle& style)
{
    int baseline = 0;
    const cv::Size text_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, style.font_scale,
                        style.font_thickness, &baseline);
    const int padding = 3;
    int left = std::max(0, x);
    int top = baseline_y - text_size.height - padding;
    const int max_width = std::max(0, image.cols - left);
    const int width = std::min(text_size.width + 2 * padding, max_width);
    const int height = std::min(text_size.height + baseline + 2 * padding,
                                std::max(0, image.rows - top));
    if (width > 0 && height > 0) {
        cv::rectangle(image, cv::Rect(left, std::max(0, top), width, height),
                      color, cv::FILLED);
    }
    const int min_text_y = text_size.height + padding;
    const int max_text_y = std::max(min_text_y, image.rows - padding);
    const int text_y = std::clamp(baseline_y, min_text_y, max_text_y);
    cv::putText(image, text, cv::Point(left + padding, text_y),
                cv::FONT_HERSHEY_SIMPLEX, style.font_scale,
                cv::Scalar(255, 255, 255), style.font_thickness, cv::LINE_8);
}

}  // namespace

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

    // Detections are only drawn for completed quality decisions. A failed
    // frame shows ERROR only and must never imply an OK/NG result.
    if (!input.has_error) {
        for (const Detection& detection : input.detections) {
            cv::Rect box;
            if (!normalizedBox(detection, annotated.cols, annotated.rows,
                               box)) {
                continue;
            }
            cv::rectangle(annotated, box, color, style.box_thickness,
                          cv::LINE_8);
            const std::string label = confidenceLabel(detection);
            int baseline = 0;
            const cv::Size text_size = cv::getTextSize(
                label, cv::FONT_HERSHEY_SIMPLEX, style.font_scale,
                style.font_thickness, &baseline);
            // Keep the label background fully above the box when possible;
            // otherwise place it just inside the top edge, leaving the box
            // outline itself unobscured.
            int label_y = box.y - baseline - 4;
            if (label_y - text_size.height - 3 < 0) {
                label_y = box.y + text_size.height + 4;
            }
            drawLabel(annotated, label, box.x, label_y, color, style);
        }
    }

    const std::string status = statusText(input);
    int baseline = 0;
    const cv::Size text_size =
        cv::getTextSize(status, cv::FONT_HERSHEY_SIMPLEX,
                        style.font_scale * 1.2, style.font_thickness + 1,
                        &baseline);
    const int banner_width =
        std::min(annotated.cols,
                 text_size.width + 2 * kStatusMargin);
    const int banner_height =
        std::min(annotated.rows,
                 text_size.height + baseline + 2 * kStatusPadding);
    if (banner_width > 0 && banner_height > 0) {
        cv::rectangle(annotated, cv::Rect(0, 0, banner_width, banner_height),
                      color, cv::FILLED);
        cv::putText(annotated, status,
                    cv::Point(kStatusMargin, banner_height - kStatusPadding),
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

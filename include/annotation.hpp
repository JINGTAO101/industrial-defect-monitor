#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "postprocess.hpp"

namespace edge_ai {

// Everything the annotation layer needs for one frame. Detections are already
// in original-image coordinates (postprocess un-letterboxes them), so no
// scaling is applied here.
struct AnnotationInput {
    bool has_error = false;
    std::string error;
    bool quality_available = false;
    bool is_ng = false;
    std::vector<Detection> detections;
};

struct AnnotationStyle {
    cv::Scalar ok_color{0, 200, 0};
    cv::Scalar ng_color{0, 0, 255};
    cv::Scalar error_color{0, 165, 255};
    double font_scale = 0.6;
    int font_thickness = 2;
    int box_thickness = 2;
};

// Returns a new BGR image with boxes/labels and an OK/NG/ERROR banner. The
// input image is never modified. Failed frames draw only ERROR; they never
// draw a quality verdict or detections.
cv::Mat annotateFrame(const cv::Mat& image,
                      const AnnotationInput& input,
                      const AnnotationStyle& style = {});

// Writes a BGR image, creating the parent directory when needed. Kept in this
// module so single-image saving and preview use the same annotation path.
bool writeAnnotatedImage(const cv::Mat& annotated,
                         const std::string& path,
                         std::string& error);

}  // namespace edge_ai

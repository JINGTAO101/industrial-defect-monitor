#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "image_source.hpp"
#include "model_runner.hpp"
#include "quality.hpp"

namespace edge_ai {

struct FrameResult {
    std::string source;
    FrameTimestamp timestamp{};
    bool inference_ok = false;
    std::string error;
    int width = 0;
    int height = 0;
    LetterboxInfo letterbox;
    std::vector<Detection> detections;
    std::vector<std::string> anomalies;
    bool quality_available = false;
    QualityResult quality;
};

struct BatchResult {
    std::vector<FrameResult> frames;
    std::size_t ok_count = 0;
    std::size_t ng_count = 0;
    std::size_t error_count = 0;
    std::size_t skipped_unsupported = 0;
};

// Consumes any ImageSource implementation. Per-frame errors (unreadable image,
// inference failure, etc.) are recorded and processing continues; the caller
// decides the process exit code. An invalid confidence threshold is fatal and
// returns false without processing frames.
bool processFrames(
    ImageSource& source,
    const ModelRunner& runner,
    const PostprocessOptions& options,
    BatchResult& result,
    std::string& fatal_error);

}  // namespace edge_ai

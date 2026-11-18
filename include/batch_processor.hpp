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
    std::size_t index = 0;
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

// Consumes any ImageSource implementation. SourceStatus::Error (unreadable
// image, per-frame inference failure, etc.) is recorded and processing
// continues. An exception escaping ImageSource::next() is treated as a fatal
// source failure and returns false immediately, because the source may not
// have advanced. An invalid confidence threshold is also fatal.
bool processFrames(
    ImageSource& source,
    const ModelRunner& runner,
    const PostprocessOptions& options,
    BatchResult& result,
    std::string& fatal_error);

}  // namespace edge_ai

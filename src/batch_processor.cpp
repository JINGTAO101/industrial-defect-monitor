#include "batch_processor.hpp"

#include <cmath>
#include <exception>
#include <utility>

namespace edge_ai {

bool processFrames(
    ImageSource& source,
    const ModelRunner& runner,
    const PostprocessOptions& options,
    BatchResult& result,
    std::string& fatal_error)
{
    fatal_error.clear();
    result = BatchResult{};
    if (!std::isfinite(options.conf_threshold) ||
        !(options.conf_threshold > 0.0f && options.conf_threshold <= 1.0f)) {
        fatal_error = "confidence threshold must be finite and in (0, 1]";
        return false;
    }

    while (true) {
        Frame frame;
        std::string error;
        SourceStatus status = SourceStatus::End;
        try {
            status = source.next(frame, error);
        } catch (const std::exception& exception) {
            // An escaping source exception means we cannot know whether the
            // source advanced. Continuing could loop forever and grow the
            // result list. Normal image failures must use SourceStatus::Error,
            // which is handled below and does continue.
            fatal_error = "image source exception";
            if (!frame.source.empty()) {
                fatal_error += " at '" + frame.source + "'";
            }
            fatal_error += ": " + std::string(exception.what());
            return false;
        }
        if (status == SourceStatus::End) {
            break;
        }

        FrameResult frame_result;
        frame_result.source =
            frame.source.empty() ? "<unknown source>" : frame.source;
        frame_result.index = frame.index;
        frame_result.timestamp = frame.timestamp;
        if (status == SourceStatus::Error) {
            frame_result.error = error;
            result.frames.push_back(std::move(frame_result));
            ++result.error_count;
            continue;
        }

        frame_result.width = frame.image.cols;
        frame_result.height = frame.image.rows;
        try {
            const PreprocessedInput input = runner.preprocess(frame.image);
            frame_result.letterbox = input.letterbox;
            const PostprocessResult postprocessed = runner.infer(input, options);
            if (!postprocessed.error.empty()) {
                frame_result.error = postprocessed.error;
            } else {
                QualityResult quality;
                std::string quality_error;
                if (!evaluateQuality(postprocessed.detections,
                                     options.conf_threshold, quality,
                                     quality_error)) {
                    frame_result.error = quality_error;
                } else {
                    frame_result.inference_ok = true;
                    frame_result.detections = postprocessed.detections;
                    frame_result.anomalies = postprocessed.anomalies;
                    frame_result.quality_available = true;
                    frame_result.quality = quality;
                }
            }
        } catch (const std::exception& exception) {
            frame_result.error =
                std::string("frame inference failed: ") + exception.what();
        }

        if (frame_result.error.empty()) {
            if (frame_result.quality.is_ng) {
                ++result.ng_count;
            } else {
                ++result.ok_count;
            }
        } else {
            ++result.error_count;
        }
        result.frames.push_back(std::move(frame_result));
    }

    result.skipped_unsupported = source.skippedUnsupported();
    return true;
}

}  // namespace edge_ai

#include <chrono>
#include <exception>
#include <iostream>
#include <string>
#include <utility>

#include <opencv2/imgcodecs.hpp>

#include "batch_processor.hpp"
#include "model_runner.hpp"

#ifndef EDGE_AI_FIXTURE_DIR
#define EDGE_AI_FIXTURE_DIR "tests/fixtures"
#endif

namespace {

// A normal bad-frame source: returns SourceStatus::Error once, then a valid
// frame. This must continue and produce one error plus one OK result.
class ErrorThenFrameSource final : public edge_ai::ImageSource {
public:
    explicit ErrorThenFrameSource(std::string image_path)
        : image_path_(std::move(image_path))
    {
    }

    edge_ai::SourceStatus next(edge_ai::Frame& frame,
                               std::string& error) override
    {
        if (step_ == 0) {
            ++step_;
            frame.source = "bad-frame";
            error = "simulated bad image";
            return edge_ai::SourceStatus::Error;
        }
        if (step_ == 1) {
            ++step_;
            cv::Mat image = cv::imread(image_path_, cv::IMREAD_COLOR);
            if (image.empty()) {
                error = "failed to read fixture: " + image_path_;
                return edge_ai::SourceStatus::Error;
            }
            frame.image = image;
            frame.source = image_path_;
            frame.index = 1;
            frame.timestamp = std::chrono::system_clock::now();
            return edge_ai::SourceStatus::Frame;
        }
        return edge_ai::SourceStatus::End;
    }

private:
    std::string image_path_;
    int step_ = 0;
};

// A broken source that throws before advancing. processFrames must stop after
// the first exception instead of retrying forever.
class AlwaysThrowingSource final : public edge_ai::ImageSource {
public:
    edge_ai::SourceStatus next(edge_ai::Frame& frame,
                               std::string&) override
    {
        ++attempts_;
        frame.source = "always-throwing";
        throw std::runtime_error("persistent source exception");
    }

    int attempts() const { return attempts_; }

private:
    int attempts_ = 0;
};

void print_batch(const edge_ai::BatchResult& batch)
{
    for (const auto& frame : batch.frames) {
        std::cerr << "  frame " << frame.source
                  << " error=" << frame.error << "\n";
    }
}

}  // namespace

int main()
{
    const std::string fixture_dir = EDGE_AI_FIXTURE_DIR;
    std::string error;
    const auto runner = edge_ai::ModelRunner::load(
        fixture_dir + "/zero_output.onnx", error);
    if (!runner) {
        std::cerr << "cannot load zero_output.onnx: " << error << "\n";
        return 1;
    }
    const edge_ai::PostprocessOptions options{};

    ErrorThenFrameSource error_then_frame(
        fixture_dir + "/non_square_320x200.png");
    edge_ai::BatchResult batch;
    if (!edge_ai::processFrames(error_then_frame, *runner, options, batch,
                                error)) {
        std::cerr << "processFrames unexpectedly failed: " << error << "\n";
        return 1;
    }
    bool ok = batch.frames.size() == 2 && batch.error_count == 1 &&
              batch.ok_count == 1 && batch.ng_count == 0;
    if (ok) {
        ok = batch.frames[0].error.find("simulated bad image") !=
                 std::string::npos &&
             !batch.frames[0].quality_available;
    }
    if (ok) {
        ok = batch.frames[1].quality_available &&
             batch.frames[1].inference_ok && !batch.frames[1].quality.is_ng &&
             batch.frames[1].timestamp != edge_ai::FrameTimestamp{};
    }
    if (!ok) {
        std::cerr << "unexpected SourceStatus::Error batch result: frames="
                  << batch.frames.size() << " errors=" << batch.error_count
                  << " ok=" << batch.ok_count << " ng=" << batch.ng_count
                  << "\n";
        print_batch(batch);
        return 1;
    }

    AlwaysThrowingSource throwing;
    edge_ai::BatchResult throwing_batch;
    std::string fatal_error;
    const bool processed = edge_ai::processFrames(
        throwing, *runner, options, throwing_batch, fatal_error);
    if (processed) {
        std::cerr << "persistently throwing source was not treated as fatal\n";
        return 1;
    }
    if (fatal_error.find("image source exception") == std::string::npos ||
        fatal_error.find("persistent source exception") == std::string::npos) {
        std::cerr << "fatal source error is not diagnostic: " << fatal_error
                  << "\n";
        return 1;
    }
    if (throwing.attempts() != 1 || !throwing_batch.frames.empty()) {
        std::cerr << "persistently throwing source was retried or recorded:\n"
                  << "  attempts=" << throwing.attempts()
                  << " frames=" << throwing_batch.frames.size() << "\n";
        return 1;
    }

    std::cout << "SourceStatus::Error continued; persistent source exception "
                 "stopped after one attempt\n";
    return 0;
}

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

class ThrowingSource final : public edge_ai::ImageSource {
public:
    explicit ThrowingSource(std::string image_path)
        : image_path_(std::move(image_path))
    {
    }

    edge_ai::SourceStatus next(edge_ai::Frame& frame,
                               std::string& error) override
    {
        if (step_ == 0) {
            ++step_;
            frame.source = "throwing-source";
            throw std::runtime_error("simulated image source exception");
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

    ThrowingSource source(fixture_dir + "/non_square_320x200.png");
    edge_ai::BatchResult batch;
    const edge_ai::PostprocessOptions options{};
    if (!edge_ai::processFrames(source, *runner, options, batch, error)) {
        std::cerr << "processFrames failed: " << error << "\n";
        return 1;
    }

    bool ok = batch.frames.size() == 2 && batch.error_count == 1 &&
              batch.ok_count == 1 && batch.ng_count == 0;
    if (ok) {
        ok = batch.frames[0].error.find("simulated image source exception") !=
             std::string::npos;
    }
    if (ok) {
        ok = !batch.frames[0].quality_available &&
             batch.frames[1].quality_available &&
             batch.frames[1].inference_ok && !batch.frames[1].quality.is_ng &&
             batch.frames[1].timestamp != edge_ai::FrameTimestamp{};
    }
    if (!ok) {
        std::cerr << "unexpected batch result: frames=" << batch.frames.size()
                  << " errors=" << batch.error_count
                  << " ok=" << batch.ok_count << " ng=" << batch.ng_count
                  << "\n";
        for (const auto& frame : batch.frames) {
            std::cerr << "  frame " << frame.source
                      << " error=" << frame.error << "\n";
        }
        return 1;
    }

    std::cout << "batch source exception was contained; next frame processed\n";
    return 0;
}

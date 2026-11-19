#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "image_preprocess.hpp"
#include "camera_source.hpp"
#include "image_source.hpp"
#include "model_contract.hpp"
#include "postprocess.hpp"
#include "quality.hpp"

#ifndef EDGE_AI_FIXTURE_DIR
#define EDGE_AI_FIXTURE_DIR "tests/fixtures"
#endif

namespace {

using TestFunction = bool (*)(std::string&);

struct TestCase {
    const char* name;
    TestFunction function;
};

bool near(float actual, float expected, float tolerance = 1e-5f)
{
    return std::fabs(actual - expected) <= tolerance;
}

bool expect(bool condition, const std::string& message, std::string& error)
{
    if (!condition) {
        error = message;
        return false;
    }
    return true;
}

edge_ai::ModelContract validContract()
{
    edge_ai::ModelContract contract;
    contract.input_count = 1;
    contract.output_count = 1;
    contract.input = {"images", {1, 3, 640, 640}, "tensor(float)"};
    contract.output = {"output0", {1, 10, 8400}, "tensor(float)"};
    contract.class_names = {
        {0, "crazing"},       {1, "inclusion"}, {2, "patches"},
        {3, "pitted_surface"}, {4, "rolled-in_scale"}, {5, "scratches"}};
    return contract;
}

std::vector<float> emptyOutput()
{
    return std::vector<float>(
        static_cast<std::size_t>(edge_ai::kOutputChannels) *
            edge_ai::kNumPredictions,
        0.0f);
}

void setPrediction(
    std::vector<float>& output,
    std::size_t index,
    float cx,
    float cy,
    float width,
    float height,
    int class_id,
    float score)
{
    output[0 * edge_ai::kNumPredictions + index] = cx;
    output[1 * edge_ai::kNumPredictions + index] = cy;
    output[2 * edge_ai::kNumPredictions + index] = width;
    output[3 * edge_ai::kNumPredictions + index] = height;
    output[(4 + static_cast<std::size_t>(class_id)) *
               edge_ai::kNumPredictions +
           index] = score;
}

edge_ai::LetterboxInfo identityLetterbox()
{
    return edge_ai::LetterboxInfo{1.0, 0, 0, 640, 640};
}

edge_ai::Detection detectionWithConfidence(float confidence)
{
    edge_ai::Detection detection;
    detection.class_id = 0;
    detection.class_name = "crazing";
    detection.confidence = confidence;
    detection.x1 = 10.0f;
    detection.y1 = 10.0f;
    detection.x2 = 20.0f;
    detection.y2 = 20.0f;
    return detection;
}

std::string fixturePath(const std::string& name)
{
    return std::string(EDGE_AI_FIXTURE_DIR) + "/" + name;
}

bool testPreprocessNonSquare(std::string& error)
{
    cv::Mat landscape(200, 320, CV_8UC3, cv::Scalar(10, 20, 30));
    const edge_ai::PreprocessedInput landscape_input =
        edge_ai::preprocessImage(landscape, 640);

    if (!expect(landscape_input.data.size() == 3u * 640u * 640u,
                "landscape tensor size is not 3*640*640", error) ||
        !expect(near(static_cast<float>(landscape_input.letterbox.ratio), 2.0f, 1e-6f),
                "landscape ratio is not 2.0", error) ||
        !expect(landscape_input.letterbox.pad_left == 0,
                "landscape pad_left is not 0", error) ||
        !expect(landscape_input.letterbox.pad_top == 120,
                "landscape pad_top is not 120", error)) {
        return false;
    }

    const std::size_t plane = 640u * 640u;
    const float pad_value = static_cast<float>(114.0 / 255.0);
    const float b_value = static_cast<float>(10.0 / 255.0);
    const float g_value = static_cast<float>(20.0 / 255.0);
    if (!expect(near(landscape_input.data[0], pad_value),
                "top-left pad value is not 114/255", error) ||
        !expect(near(landscape_input.data[120 * 640], b_value),
                "first resized B value is not 10/255", error) ||
        !expect(near(landscape_input.data[plane + 120 * 640], g_value),
                "first resized G value is not 20/255", error) ||
        !expect(near(landscape_input.data[639 * 640], pad_value),
                "bottom pad value is not 114/255", error)) {
        return false;
    }

    cv::Mat portrait(320, 200, CV_8UC3, cv::Scalar(10, 20, 30));
    const edge_ai::PreprocessedInput portrait_input =
        edge_ai::preprocessImage(portrait, 640);
    if (!expect(portrait_input.letterbox.pad_left == 120,
                "portrait pad_left is not 120", error) ||
        !expect(portrait_input.letterbox.pad_top == 0,
                "portrait pad_top is not 0", error) ||
        !expect(near(portrait_input.data[0], pad_value),
                "left pad value is not 114/255", error) ||
        !expect(near(portrait_input.data[120], b_value),
                "first portrait content value is not 10/255", error)) {
        return false;
    }
    return true;
}

bool testPostprocessEmpty(std::string& error)
{
    const edge_ai::PostprocessResult result = edge_ai::postprocess(
        emptyOutput(), validContract().class_names, identityLetterbox(),
        edge_ai::PostprocessOptions{});
    return expect(result.error.empty(), "empty output returned an error: " + result.error, error) &&
           expect(result.detections.empty(), "empty output produced detections", error);
}

bool testPostprocessNmsSameClass(std::string& error)
{
    std::vector<float> output = emptyOutput();
    setPrediction(output, 0, 100, 100, 50, 50, 0, 0.9f);
    setPrediction(output, 1, 105, 105, 50, 50, 0, 0.8f);
    setPrediction(output, 2, 300, 300, 50, 50, 0, 0.7f);

    const edge_ai::PostprocessResult result = edge_ai::postprocess(
        output, validContract().class_names, identityLetterbox(),
        edge_ai::PostprocessOptions{});
    if (!expect(result.error.empty(), "same-class NMS returned an error: " + result.error, error) ||
        !expect(result.detections.size() == 2,
                "same-class NMS expected 2 detections, got " +
                    std::to_string(result.detections.size()),
                error)) {
        return false;
    }
    return expect(result.detections[0].confidence == 0.9f,
                  "same-class NMS did not keep the highest score first", error) &&
           expect(result.detections[1].confidence == 0.7f,
                  "same-class NMS did not keep the distant box", error);
}

bool testPostprocessNmsCrossClass(std::string& error)
{
    std::vector<float> output = emptyOutput();
    setPrediction(output, 0, 100, 100, 50, 50, 0, 0.9f);
    setPrediction(output, 1, 100, 100, 50, 50, 1, 0.8f);

    const edge_ai::PostprocessResult result = edge_ai::postprocess(
        output, validContract().class_names, identityLetterbox(),
        edge_ai::PostprocessOptions{});
    return expect(result.error.empty(), "cross-class NMS returned an error: " + result.error, error) &&
           expect(result.detections.size() == 2,
                  "cross-class NMS must not suppress a different class", error);
}

bool testPostprocessCoordinateRestore(std::string& error)
{
    std::vector<float> output = emptyOutput();
    setPrediction(output, 0, 190, 100, 100, 120, 0, 0.9f);
    const edge_ai::LetterboxInfo letterbox{2.0, 120, 0, 200, 320};

    const edge_ai::PostprocessResult result = edge_ai::postprocess(
        output, validContract().class_names, letterbox,
        edge_ai::PostprocessOptions{});
    if (!expect(result.error.empty(), "coordinate test returned an error: " + result.error, error) ||
        !expect(result.detections.size() == 1, "coordinate test expected 1 detection", error)) {
        return false;
    }
    const auto& detection = result.detections[0];
    return expect(near(detection.x1, 10.0f, 1e-4f), "x1 restoration mismatch", error) &&
           expect(near(detection.y1, 20.0f, 1e-4f), "y1 restoration mismatch", error) &&
           expect(near(detection.x2, 60.0f, 1e-4f), "x2 restoration mismatch", error) &&
           expect(near(detection.y2, 80.0f, 1e-4f), "y2 restoration mismatch", error);
}

bool testPostprocessNonFinite(std::string& error)
{
    std::vector<float> output = emptyOutput();
    output[0] = std::nanf("");
    const edge_ai::PostprocessResult result = edge_ai::postprocess(
        output, validContract().class_names, identityLetterbox(),
        edge_ai::PostprocessOptions{});
    return expect(!result.error.empty(), "non-finite output was not rejected", error) &&
           expect(result.error.find("non-finite") != std::string::npos,
                  "non-finite error is not diagnostic: " + result.error, error) &&
           expect(result.detections.empty(), "non-finite output produced detections", error);
}

bool testPostprocessScoreRange(std::string& error)
{
    {
        std::vector<float> output = emptyOutput();
        setPrediction(output, 0, 100, 100, 50, 50, 0, 1.5f);
        const edge_ai::PostprocessResult result = edge_ai::postprocess(
            output, validContract().class_names, identityLetterbox(),
            edge_ai::PostprocessOptions{});
        if (!expect(!result.error.empty(),
                    "score > 1 was not rejected", error) ||
            !expect(result.error.find("score") != std::string::npos &&
                        result.error.find("range") != std::string::npos,
                    "score > 1 error is not diagnostic: " + result.error, error) ||
            !expect(result.detections.empty(),
                    "score > 1 produced a detection", error)) {
            return false;
        }
    }
    {
        std::vector<float> output = emptyOutput();
        setPrediction(output, 0, 100, 100, 50, 50, 0, -0.1f);
        const edge_ai::PostprocessResult result = edge_ai::postprocess(
            output, validContract().class_names, identityLetterbox(),
            edge_ai::PostprocessOptions{});
        if (!expect(!result.error.empty(),
                    "negative score was not rejected", error) ||
            !expect(result.detections.empty(),
                    "negative score produced a detection", error)) {
            return false;
        }
    }
    {
        std::vector<float> output = emptyOutput();
        setPrediction(output, 0, 100, 100, 50, 50, 0, 1.0f);
        const edge_ai::PostprocessResult result = edge_ai::postprocess(
            output, validContract().class_names, identityLetterbox(),
            edge_ai::PostprocessOptions{});
        if (!expect(result.error.empty(),
                    "score 1.0 boundary was rejected: " + result.error, error) ||
            !expect(result.detections.size() == 1,
                    "score 1.0 boundary should produce one detection", error)) {
            return false;
        }
    }
    {
        std::vector<float> output = emptyOutput();
        setPrediction(output, 0, 100, 100, 50, 50, 0, 0.0f);
        const edge_ai::PostprocessResult result = edge_ai::postprocess(
            output, validContract().class_names, identityLetterbox(),
            edge_ai::PostprocessOptions{});
        if (!expect(result.error.empty(),
                    "score 0.0 boundary was rejected: " + result.error, error) ||
            !expect(result.detections.empty(),
                    "score 0.0 boundary should be below the confidence threshold",
                    error)) {
            return false;
        }
    }
    return true;
}

bool testPostprocessShapeMismatch(std::string& error)
{
    const std::vector<float> output(100, 0.0f);
    const edge_ai::PostprocessResult result = edge_ai::postprocess(
        output, validContract().class_names, identityLetterbox(),
        edge_ai::PostprocessOptions{});
    return expect(!result.error.empty(), "wrong output size was not rejected", error) &&
           expect(result.error.find("size mismatch") != std::string::npos,
                  "size mismatch error is not diagnostic: " + result.error, error);
}

bool testClassMappingUnknown(std::string& error)
{
    const std::map<int, std::string> class_names = {{0, "crazing"}};
    std::vector<std::string> anomalies;
    const std::string name = edge_ai::classNameFor(7, class_names, anomalies);
    return expect(name == "Unknown", "unknown class was not mapped to Unknown", error) &&
           expect(anomalies.size() == 1, "unknown class did not record an anomaly", error);
}

bool testContractValidation(std::string& error)
{
    if (!expect(edge_ai::validateModelContract(validContract()).empty(),
                "valid contract was rejected", error)) {
        return false;
    }

    edge_ai::ModelContract bad_input = validContract();
    bad_input.input.shape = {1, 3, 320, 320};
    const std::string input_error = edge_ai::validateModelContract(bad_input);
    if (!expect(input_error.find("input shape") != std::string::npos,
                "input shape mismatch not reported: " + input_error, error)) {
        return false;
    }

    edge_ai::ModelContract bad_output = validContract();
    bad_output.output.shape = {1, 10, 100};
    const std::string output_error = edge_ai::validateModelContract(bad_output);
    if (!expect(output_error.find("output shape") != std::string::npos,
                "output shape mismatch not reported: " + output_error, error)) {
        return false;
    }

    edge_ai::ModelContract bad_name = validContract();
    bad_name.input.name = "input";
    if (!expect(edge_ai::validateModelContract(bad_name).find("input name") !=
                    std::string::npos,
                "input name mismatch not reported", error)) {
        return false;
    }

    edge_ai::ModelContract bad_class = validContract();
    bad_class.class_names[3] = "wrong";
    if (!expect(edge_ai::validateModelContract(bad_class).find("class id 3") !=
                    std::string::npos,
                "class name mismatch not reported", error)) {
        return false;
    }

    edge_ai::ModelContract bad_count = validContract();
    bad_count.input_count = 2;
    if (!expect(edge_ai::validateModelContract(bad_count).find("input count") !=
                    std::string::npos,
                "input count mismatch not reported", error)) {
        return false;
    }

    edge_ai::ModelContract bad_type = validContract();
    bad_type.output.type = "tensor(double)";
    if (!expect(edge_ai::validateModelContract(bad_type).find("output type") !=
                    std::string::npos,
                "output type mismatch not reported", error)) {
        return false;
    }
    return true;
}

bool testQualityNoDefects(std::string& error)
{
    edge_ai::QualityResult result;
    std::string quality_error;
    if (!expect(edge_ai::evaluateQuality({}, 0.5f, result, quality_error),
                "empty detections were rejected: " + quality_error, error)) {
        return false;
    }
    return expect(!result.is_ng, "empty detections must be OK", error) &&
           expect(result.defects.empty(), "OK result has defects", error) &&
           expect(result.total_detections == 0,
                  "OK result has a non-zero detection count", error);
}

bool testQualitySingleDefect(std::string& error)
{
    edge_ai::QualityResult result;
    std::string quality_error;
    if (!expect(edge_ai::evaluateQuality(
                    {detectionWithConfidence(0.9f)}, 0.5f, result,
                    quality_error),
                "single defect was rejected: " + quality_error, error)) {
        return false;
    }
    return expect(result.is_ng, "single qualifying defect must be NG", error) &&
           expect(result.defects.size() == 1,
                  "NG result must contain the qualifying defect", error);
}

bool testQualityBelowThreshold(std::string& error)
{
    edge_ai::QualityResult result;
    std::string quality_error;
    if (!expect(edge_ai::evaluateQuality(
                    {detectionWithConfidence(0.4f)}, 0.5f, result,
                    quality_error),
                "below-threshold detection was rejected: " + quality_error,
                error)) {
        return false;
    }
    return expect(!result.is_ng,
                  "below-threshold detection must be OK", error) &&
           expect(result.total_detections == 1,
                  "below-threshold total count is wrong", error) &&
           expect(result.defects.empty(),
                  "below-threshold detection must not qualify", error);
}

bool testQualityMultipleDefects(std::string& error)
{
    edge_ai::QualityResult result;
    std::string quality_error;
    if (!expect(edge_ai::evaluateQuality(
                    {detectionWithConfidence(0.9f),
                     detectionWithConfidence(0.8f)},
                    0.5f, result, quality_error),
                "multiple defects were rejected: " + quality_error, error)) {
        return false;
    }
    return expect(result.is_ng, "multiple qualifying defects must be NG", error) &&
           expect(result.defects.size() == 2,
                  "NG result must contain both qualifying defects", error);
}

bool testQualityBoundary(std::string& error)
{
    edge_ai::QualityResult result;
    std::string quality_error;
    if (!expect(edge_ai::evaluateQuality(
                    {detectionWithConfidence(0.5f)}, 0.5f, result,
                    quality_error),
                "boundary detection was rejected: " + quality_error, error)) {
        return false;
    }
    return expect(!result.is_ng,
                  "confidence exactly at threshold must be OK (strict >)", error) &&
           expect(result.defects.empty(),
                  "boundary detection must not qualify", error);
}

bool testQualityInvalidThreshold(std::string& error)
{
    const std::vector<float> invalid = {
        0.0f, -0.1f, 1.1f, std::nanf(""),
        std::numeric_limits<float>::infinity()};
    for (const float threshold : invalid) {
        edge_ai::QualityResult result;
        std::string quality_error;
        if (!expect(!edge_ai::evaluateQuality(
                        {detectionWithConfidence(0.9f)}, threshold, result,
                        quality_error),
                    "invalid threshold was accepted: " +
                        std::to_string(threshold),
                    error) ||
            !expect(quality_error.find("threshold") != std::string::npos,
                    "invalid threshold error is not diagnostic: " +
                        quality_error,
                    error) ||
            !expect(!result.is_ng && result.defects.empty(),
                    "invalid threshold produced a decision", error)) {
            return false;
        }
    }
    return true;
}

bool testImageSourceSingle(std::string& error)
{
    std::string source_error;
    edge_ai::SingleImageSource valid(
        fixturePath("non_square_320x200.png"), source_error);
    if (!expect(source_error.empty(),
                "valid single source rejected: " + source_error, error)) {
        return false;
    }
    edge_ai::Frame frame;
    const auto timestamp_before = std::chrono::system_clock::now();
    const edge_ai::SourceStatus frame_status = valid.next(frame, source_error);
    const auto timestamp_after = std::chrono::system_clock::now();
    if (!expect(frame_status == edge_ai::SourceStatus::Frame,
                "valid single source did not return a frame: " + source_error,
                error) ||
        !expect(!frame.image.empty() && frame.image.cols == 320,
                "valid single source loaded the wrong image", error) ||
        !expect(frame.timestamp >= timestamp_before &&
                    frame.timestamp <= timestamp_after,
                "single source frame has no valid read timestamp", error) ||
        !expect(valid.next(frame, source_error) == edge_ai::SourceStatus::End,
                "valid single source did not end", error)) {
        return false;
    }

    edge_ai::SingleImageSource missing(
        "/tmp/edge_ai_task3_does_not_exist.png", source_error);
    if (!expect(!source_error.empty() &&
                    source_error.find("not found") != std::string::npos,
                "missing image was not rejected: " + source_error, error)) {
        return false;
    }

    edge_ai::SingleImageSource unsupported(
        fixturePath("README.md"), source_error);
    if (!expect(!source_error.empty() &&
                    source_error.find("unsupported") != std::string::npos,
                "unsupported extension was not rejected: " + source_error,
                error)) {
        return false;
    }

    const std::filesystem::path corrupt_path =
        std::filesystem::temp_directory_path() /
        "edge_ai_task3_corrupt.jpg";
    {
        std::ofstream corrupt(corrupt_path, std::ios::binary);
        corrupt << "not an image";
    }
    edge_ai::SingleImageSource corrupt(corrupt_path.string(), source_error);
    if (!expect(source_error.empty(),
                "corrupt file was rejected too early: " + source_error,
                error)) {
        std::filesystem::remove(corrupt_path);
        return false;
    }
    const edge_ai::SourceStatus corrupt_status =
        corrupt.next(frame, source_error);
    const bool corrupt_ok =
        expect(corrupt_status == edge_ai::SourceStatus::Error,
               "corrupt image did not return Error", error) &&
        expect(source_error.find("failed to read image") != std::string::npos,
               "corrupt image error is not diagnostic: " + source_error,
               error) &&
        expect(corrupt.next(frame, source_error) == edge_ai::SourceStatus::End,
               "corrupt single source did not end", error);
    std::filesystem::remove(corrupt_path);
    return corrupt_ok;
}

bool testImageSourceDirectory(std::string& error)
{
    const std::filesystem::path test_dir =
        std::filesystem::temp_directory_path() /
        "edge_ai_task3_source_dir";
    std::error_code filesystem_error;
    std::filesystem::remove_all(test_dir, filesystem_error);
    std::filesystem::create_directories(test_dir, filesystem_error);
    std::filesystem::copy_file(
        fixturePath("non_square_320x200.png"), test_dir / "a.png",
        std::filesystem::copy_options::overwrite_existing, filesystem_error);
    {
        std::ofstream corrupt(test_dir / "b.jpg", std::ios::binary);
        corrupt << "not an image";
    }
    {
        std::ofstream unsupported(test_dir / "c.txt", std::ios::binary);
        unsupported << "ignored";
    }

    std::string source_error;
    edge_ai::DirectoryImageSource source(test_dir.string(), source_error);
    if (!expect(source_error.empty(),
                "valid directory source rejected: " + source_error, error) ||
        !expect(source.skippedUnsupported() == 1,
                "directory source did not count the unsupported file", error)) {
        std::filesystem::remove_all(test_dir);
        return false;
    }

    edge_ai::Frame frame;
    const auto directory_timestamp_before = std::chrono::system_clock::now();
    const edge_ai::SourceStatus directory_frame_status =
        source.next(frame, source_error);
    const auto directory_timestamp_after = std::chrono::system_clock::now();
    if (!expect(directory_frame_status == edge_ai::SourceStatus::Frame,
                "directory source did not return the valid frame: " +
                    source_error,
                error) ||
        !expect(frame.source.find("a.png") != std::string::npos,
                "directory source returned the wrong order", error) ||
        !expect(frame.timestamp >= directory_timestamp_before &&
                    frame.timestamp <= directory_timestamp_after,
                "directory source frame has no valid read timestamp", error) ||
        !expect(source.next(frame, source_error) ==
                    edge_ai::SourceStatus::Error,
                "directory source did not report the corrupt frame", error) ||
        !expect(source_error.find("failed to read image") != std::string::npos,
                "directory corrupt error is not diagnostic: " + source_error,
                error) ||
        !expect(source.next(frame, source_error) == edge_ai::SourceStatus::End,
                "directory source did not continue to End after a bad file",
                error)) {
        std::filesystem::remove_all(test_dir);
        return false;
    }
    std::filesystem::remove_all(test_dir);

    edge_ai::DirectoryImageSource missing(
        (std::filesystem::temp_directory_path() /
         "edge_ai_task3_missing_dir").string(),
        source_error);
    if (!expect(!source_error.empty() &&
                    source_error.find("not found") != std::string::npos,
                "missing directory was not rejected: " + source_error, error)) {
        return false;
    }

    const std::filesystem::path empty_dir =
        std::filesystem::temp_directory_path() / "edge_ai_task3_empty_dir";
    std::filesystem::remove_all(empty_dir, filesystem_error);
    std::filesystem::create_directories(empty_dir, filesystem_error);
    edge_ai::DirectoryImageSource empty(empty_dir.string(), source_error);
    std::filesystem::remove_all(empty_dir);
    if (!expect(!source_error.empty() &&
                    source_error.find("no supported images") !=
                        std::string::npos,
                "empty directory was not rejected: " + source_error, error)) {
        return false;
    }

    edge_ai::DirectoryImageSource file_source(
        fixturePath("non_square_320x200.png"), source_error);
    return expect(!source_error.empty() &&
                      source_error.find("not a directory") != std::string::npos,
                  "regular file was not rejected as a directory: " +
                      source_error,
                  error);
}

class FakeTimeoutCameraBackend final : public edge_ai::CameraBackend {
public:
    FakeTimeoutCameraBackend()
    {
        info_.device = "/dev/fake";
        info_.actual_format = "YUYV";
        info_.actual_width = 640;
        info_.actual_height = 480;
        info_.actual_fps = 30.0;
        info_.bytes_per_line = 1280;
    }

    bool nextFrame(std::vector<std::uint8_t>&, std::string& error) override
    {
        error = "camera read timeout after 100 ms";
        return false;
    }

    const edge_ai::CameraInfo& info() const override { return info_; }
    void close() noexcept override {}

private:
    edge_ai::CameraInfo info_;
};

class FakeYuyvCameraBackend final : public edge_ai::CameraBackend {
public:
    FakeYuyvCameraBackend()
    {
        info_.device = "/dev/fake-yuyv";
        info_.actual_format = "YUYV";
        info_.actual_width = 2;
        info_.actual_height = 2;
        info_.actual_fps = 30.0;
        info_.bytes_per_line = 4;
    }

    bool nextFrame(std::vector<std::uint8_t>& payload,
                   std::string& error) override
    {
        if (delivered_ >= 2) {
            error = "fake camera ended";
            return false;
        }
        payload = {0, 128, 255, 128, 0, 128, 255, 128};
        ++delivered_;
        return true;
    }

    const edge_ai::CameraInfo& info() const override { return info_; }
    void close() noexcept override {}

private:
    edge_ai::CameraInfo info_;
    int delivered_ = 0;
};

bool testCameraMissingDevice(std::string& error)
{
    edge_ai::CameraOptions options;
    options.device = "/dev/edge_ai_missing_camera";
    options.format = "YUYV";
    std::string source_error;
    edge_ai::CameraSource source(options, source_error);
    return expect(!source_error.empty() &&
                      source_error.find("not found") != std::string::npos,
                  "missing camera device was not diagnosed: " + source_error,
                  error);
}

bool testCameraTimeout(std::string& error)
{
    edge_ai::CameraOptions options;
    options.device = "/dev/fake";
    options.format = "YUYV";
    options.width = 640;
    options.height = 480;
    options.fps = 30;
    options.timeout_ms = 100;
    std::string source_error;
    edge_ai::CameraSource source(
        options, std::make_unique<FakeTimeoutCameraBackend>(), source_error);
    if (!expect(source_error.empty(),
                "fake timeout camera was rejected: " + source_error, error)) {
        return false;
    }
    edge_ai::Frame frame;
    const edge_ai::SourceStatus status = source.next(frame, source_error);
    return expect(status == edge_ai::SourceStatus::Error,
                  "timeout did not return Error", error) &&
           expect(source_error.find("timeout") != std::string::npos,
                  "timeout error is not diagnostic: " + source_error, error);
}

bool testCameraInvalidOptions(std::string& error)
{
    edge_ai::CameraOptions options;
    options.device = "/dev/fake";
    options.format = "RGB24";
    std::string source_error;
    edge_ai::CameraSource source(
        options, std::make_unique<FakeTimeoutCameraBackend>(), source_error);
    return expect(!source_error.empty() &&
                      source_error.find("unsupported camera pixel format") !=
                          std::string::npos,
                  "unsupported camera format was not rejected: " +
                      source_error,
                  error);
}

bool testCameraFrameConversion(std::string& error)
{
    edge_ai::CameraOptions options;
    options.device = "/dev/fake-yuyv";
    options.format = "YUYV";
    options.width = 2;
    options.height = 2;
    options.fps = 30;
    options.timeout_ms = 100;
    std::string source_error;
    edge_ai::CameraSource source(
        options, std::make_unique<FakeYuyvCameraBackend>(), source_error);
    if (!expect(source_error.empty(),
                "fake YUYV camera was rejected: " + source_error, error)) {
        return false;
    }

    edge_ai::Frame first;
    if (!expect(source.next(first, source_error) ==
                    edge_ai::SourceStatus::Frame,
                "first fake camera frame failed: " + source_error, error) ||
        !expect(first.index == 0, "first camera frame index is not 0", error) ||
        !expect(first.image.cols == 2 && first.image.rows == 2 &&
                    first.image.type() == CV_8UC3,
                "YUYV did not convert to 2x2 BGR", error) ||
        !expect(first.timestamp != edge_ai::FrameTimestamp{},
                "camera frame has no timestamp", error)) {
        return false;
    }

    edge_ai::Frame second;
    return expect(source.next(second, source_error) ==
                      edge_ai::SourceStatus::Frame,
                  "second fake camera frame failed: " + source_error, error) &&
           expect(second.index == 1, "second camera frame index is not 1",
                  error);
}

bool testClassMetadataParser(std::string& error)
{
    std::map<int, std::string> names;
    std::string parse_error;
    if (!expect(edge_ai::parseClassNamesMetadata(
                    "{0: 'crazing', 1: 'inclusion'}", names, parse_error),
                "valid class metadata was rejected: " + parse_error, error) ||
        !expect(names.size() == 2 && names[1] == "inclusion",
                "parsed class metadata is incorrect", error)) {
        return false;
    }

    std::map<int, std::string> duplicate;
    if (!expect(!edge_ai::parseClassNamesMetadata(
                    "{0: 'a', 0: 'b'}", duplicate, parse_error),
                "duplicate class ids were accepted", error)) {
        return false;
    }

    std::map<int, std::string> gap;
    if (!expect(!edge_ai::parseClassNamesMetadata(
                    "{0: 'a', 2: 'c'}", gap, parse_error),
                "non-contiguous class ids were accepted", error)) {
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char* argv[])
{
    const std::vector<TestCase> tests = {
        {"preprocess_non_square", testPreprocessNonSquare},
        {"postprocess_empty", testPostprocessEmpty},
        {"postprocess_nms_same_class", testPostprocessNmsSameClass},
        {"postprocess_nms_cross_class", testPostprocessNmsCrossClass},
        {"postprocess_coordinate_restore", testPostprocessCoordinateRestore},
        {"postprocess_nonfinite", testPostprocessNonFinite},
        {"postprocess_score_range", testPostprocessScoreRange},
        {"postprocess_shape_mismatch", testPostprocessShapeMismatch},
        {"class_mapping_unknown", testClassMappingUnknown},
        {"contract_validation", testContractValidation},
        {"class_metadata_parser", testClassMetadataParser},
        {"quality_no_defects", testQualityNoDefects},
        {"quality_single_defect", testQualitySingleDefect},
        {"quality_below_threshold", testQualityBelowThreshold},
        {"quality_multiple_defects", testQualityMultipleDefects},
        {"quality_boundary", testQualityBoundary},
        {"quality_invalid_threshold", testQualityInvalidThreshold},
        {"image_source_single", testImageSourceSingle},
        {"image_source_directory", testImageSourceDirectory},
        {"camera_missing_device", testCameraMissingDevice},
        {"camera_timeout", testCameraTimeout},
        {"camera_invalid_options", testCameraInvalidOptions},
        {"camera_frame_conversion", testCameraFrameConversion},
    };

    std::string filter;
    if (argc > 1) {
        filter = argv[1];
    }
    int failures = 0;
    int ran = 0;
    for (const TestCase& test : tests) {
        if (!filter.empty() && filter != test.name) {
            continue;
        }
        ++ran;
        std::string error;
        if (test.function(error)) {
            std::cout << "[PASS] " << test.name << "\n";
        } else {
            ++failures;
            std::cout << "[FAIL] " << test.name << ": " << error << "\n";
        }
    }
    if (ran == 0) {
        std::cerr << "error: unknown test filter: " << filter << "\n";
        return 2;
    }
    std::cout << ran << " test(s), " << failures << " failure(s)\n";
    return failures == 0 ? 0 : 1;
}

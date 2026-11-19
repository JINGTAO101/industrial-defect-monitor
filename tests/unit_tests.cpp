#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <thread>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <linux/videodev2.h>

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

bool testCameraBufferValidation(std::string& error)
{
    std::string buffer_error;
    if (!expect(!edge_ai::validateCameraBuffer(
                    true, 0, 4, 100, 100, buffer_error),
                "error-flagged camera buffer was accepted", error) ||
        !expect(buffer_error.find("V4L2_BUF_FLAG_ERROR") != std::string::npos,
                "error-flag diagnostic is unclear: " + buffer_error, error) ||
        !expect(!edge_ai::validateCameraBuffer(
                    false, 0, 4, 0, 100, buffer_error),
                "empty camera buffer was accepted", error) ||
        !expect(!edge_ai::validateCameraBuffer(
                    false, 0, 4, 101, 100, buffer_error),
                "overlength camera buffer was accepted", error) ||
        !expect(!edge_ai::validateCameraBuffer(
                    false, 4, 4, 100, 100, buffer_error),
                "invalid camera buffer index was accepted", error)) {
        return false;
    }
    return expect(edge_ai::validateCameraBuffer(
                      false, 0, 4, 100, 100, buffer_error),
                  "valid camera buffer was rejected: " + buffer_error, error);
}

bool testCameraBusyError(std::string& error)
{
    const std::string busy = edge_ai::cameraInitializationError(
        "camera format not supported", "/dev/video0", EBUSY);
    const std::string permission = edge_ai::cameraInitializationError(
        "camera format not supported", "/dev/video0", EACCES);
    return expect(busy.find("camera device busy") != std::string::npos,
                  "EBUSY was not classified as device busy: " + busy,
                  error) &&
           expect(permission.find("camera permission denied") !=
                      std::string::npos,
                  "EACCES was not classified as permission denied: " +
                      permission,
                  error);
}

bool testCameraReadDeadline(std::string& error)
{
    {
        edge_ai::CameraReadOps operations;
        operations.poll = [](int, int& error_number) {
            error_number = EINTR;
            return -1;
        };
        operations.dequeue = [](int&) { return 0; };
        const auto start = std::chrono::steady_clock::now();
        std::string read_error;
        const edge_ai::CameraReadResult result =
            edge_ai::waitForCameraFrame(operations, 50, read_error);
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
        if (!expect(result == edge_ai::CameraReadResult::Timeout,
                    "EINTR poll loop did not time out: " + read_error, error) ||
            !expect(read_error.find("timeout") != std::string::npos,
                    "timeout diagnostic is unclear: " + read_error, error) ||
            !expect(elapsed < 1000, "EINTR poll loop exceeded its deadline",
                    error)) {
            return false;
        }
    }
    {
        edge_ai::CameraReadOps operations;
        operations.poll = [](int, int&) { return 1; };
        operations.dequeue = [](int& error_number) {
            error_number = EAGAIN;
            return -1;
        };
        const auto start = std::chrono::steady_clock::now();
        std::string read_error;
        const edge_ai::CameraReadResult result =
            edge_ai::waitForCameraFrame(operations, 50, read_error);
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
        if (!expect(result == edge_ai::CameraReadResult::Timeout,
                    "EAGAIN dequeue loop did not time out: " + read_error,
                    error) ||
            !expect(elapsed >= 20,
                    "EAGAIN dequeue returned before the deadline", error)) {
            return false;
        }
    }
    // A single VIDIOC_DQBUF must not self-retry EINTR: when it keeps
    // interrupting, the shared deadline must expire and report timeout, never
    // deliver a frame after the budget is exhausted. Use the same one-shot
    // production helper with an injected ioctl that returns EINTR.
    {
        edge_ai::CameraReadOps operations;
        operations.poll = [](int, int&) { return 1; };
        int dequeue_calls = 0;
        v4l2_buffer buffer{};
        operations.dequeue = [&dequeue_calls, &buffer](int& error_number) {
            errno = 0;
            const int result = edge_ai::dequeueCameraBufferOnce(
                -1, VIDIOC_DQBUF, &buffer,
                [&dequeue_calls](int, unsigned long, void*) {
                    ++dequeue_calls;
                    errno = EINTR;
                    return -1;
                });
            error_number = errno;
            return result;
        };
        std::string read_error;
        const auto start = std::chrono::steady_clock::now();
        const edge_ai::CameraReadResult result =
            edge_ai::waitForCameraFrame(operations, 100, read_error);
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start)
                .count();
        if (!expect(result == edge_ai::CameraReadResult::Timeout,
                    "continuous DQBUF EINTR did not time out: " + read_error,
                    error) ||
            !expect(read_error.find("timeout") != std::string::npos,
                    "continuous DQBUF EINTR timeout is not diagnostic: " +
                        read_error,
                    error) ||
            !expect(elapsed < 500,
                    "continuous DQBUF EINTR ran far past its 100ms deadline",
                    error) ||
            !expect(dequeue_calls > 1,
                    "continuous DQBUF EINTR was not retried by the deadline loop",
                    error)) {
            return false;
        }
    }

    edge_ai::CameraReadOps operations;
    int dequeue_calls = 0;
    operations.poll = [](int, int&) { return 1; };
    operations.dequeue = [&dequeue_calls](int& error_number) {
        if (++dequeue_calls == 1) {
            error_number = EINTR;
            return -1;
        }
        return 0;
    };
    std::string read_error;
    return expect(edge_ai::waitForCameraFrame(operations, 50, read_error) ==
                      edge_ai::CameraReadResult::FrameReady,
                  "dequeue EINTR was not retried within budget: " + read_error,
                  error);
}

bool testCameraReadStop(std::string& error)
{
    std::atomic<bool> stop{false};
    edge_ai::CameraReadOps operations;
    operations.should_stop = [&stop] { return stop.load(); };
    operations.poll = [](int timeout_ms, int&) {
        std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
        return 0;
    };
    operations.dequeue = [](int&) { return -1; };

    std::thread stopper([&stop] {
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
        stop.store(true);
    });
    const auto start = std::chrono::steady_clock::now();
    std::string read_error;
    const edge_ai::CameraReadResult result =
        edge_ai::waitForCameraFrame(operations, 3000, read_error);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count();
    stopper.join();
    return expect(result == edge_ai::CameraReadResult::Stopped,
                  "stop during a frame wait did not return Stopped: " +
                      read_error,
                  error) &&
           expect(read_error.find("stopped") != std::string::npos,
                  "stopped read error is unclear: " + read_error, error) &&
           expect(elapsed < 1000,
                  "stop during a frame wait did not slice the poll timeout",
                  error);
}

bool testCameraSourceExternalStop(std::string& error)
{
    class StopAwareBackend final : public edge_ai::CameraBackend {
    public:
        StopAwareBackend()
        {
            info_.device = "/dev/stop-aware";
            info_.actual_format = "YUYV";
            info_.actual_width = 2;
            info_.actual_height = 2;
            info_.actual_fps = 30.0;
            info_.bytes_per_line = 4;
        }

        bool nextFrame(std::vector<std::uint8_t>&,
                       std::string& read_error) override
        {
            edge_ai::CameraReadOps operations;
            operations.should_stop = [this] {
                return stop_ != nullptr && stop_->load();
            };
            operations.poll = [](int timeout_ms, int&) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(timeout_ms));
                return 0;
            };
            operations.dequeue = [](int&) { return -1; };
            const edge_ai::CameraReadResult result =
                edge_ai::waitForCameraFrame(operations, 3000, read_error);
            return result == edge_ai::CameraReadResult::FrameReady;
        }

        const edge_ai::CameraInfo& info() const override { return info_; }
        void setExternalStop(const std::atomic<bool>* stop) override
        {
            stop_ = stop;
        }
        void close() noexcept override {}

    private:
        edge_ai::CameraInfo info_;
        const std::atomic<bool>* stop_ = nullptr;
    };

    edge_ai::CameraOptions options;
    options.device = "/dev/stop-aware";
    options.format = "YUYV";
    options.width = 2;
    options.height = 2;
    options.fps = 30;
    options.timeout_ms = 3000;
    std::string source_error;
    edge_ai::CameraSource source(
        options, std::make_unique<StopAwareBackend>(), source_error);
    if (!expect(source_error.empty(),
                "stop-aware camera construction failed: " + source_error,
                error)) {
        return false;
    }
    std::atomic<bool> stop{false};
    source.setExternalStop(&stop);

    edge_ai::SourceStatus status = edge_ai::SourceStatus::Error;
    const auto start = std::chrono::steady_clock::now();
    std::thread worker([&] {
        edge_ai::Frame frame;
        status = source.next(frame, source_error);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    stop.store(true);
    worker.join();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count();
    return expect(status == edge_ai::SourceStatus::End,
                  "camera source did not end on external stop", error) &&
           expect(source_error.empty(),
                  "external stop produced an error: " + source_error, error) &&
           expect(elapsed < 1000,
                  "camera source waited for the full read timeout on stop",
                  error);
}

bool testCameraColorimetry(std::string& error)
{
    return expect(edge_ai::supportedYuyvColorimetry(
                      V4L2_COLORSPACE_SRGB, V4L2_YCBCR_ENC_601,
                      V4L2_QUANTIZATION_LIM_RANGE),
                  "BT.601 limited YUYV was rejected", error) &&
           expect(edge_ai::supportedYuyvColorimetry(
                      V4L2_COLORSPACE_DEFAULT, V4L2_YCBCR_ENC_DEFAULT,
                      V4L2_QUANTIZATION_DEFAULT),
                  "default YUYV colorimetry was rejected", error) &&
           expect(!edge_ai::supportedYuyvColorimetry(
                      V4L2_COLORSPACE_SRGB, V4L2_YCBCR_ENC_601,
                      V4L2_QUANTIZATION_FULL_RANGE),
                  "full-range YUYV was accepted", error) &&
           expect(!edge_ai::supportedYuyvColorimetry(
                      V4L2_COLORSPACE_SRGB, V4L2_YCBCR_ENC_709,
                      V4L2_QUANTIZATION_LIM_RANGE),
                  "BT.709 YUYV was accepted", error) &&
           expect(!edge_ai::supportedYuyvColorimetry(
                      V4L2_COLORSPACE_JPEG, V4L2_YCBCR_ENC_601,
                      V4L2_QUANTIZATION_LIM_RANGE),
                  "JPEG colorspace YUYV was accepted", error);
}

class ScriptedCameraBackend final : public edge_ai::CameraBackend {
public:
    enum class Event { Frame, Bad, Lost };

    explicit ScriptedCameraBackend(std::vector<Event> events)
        : events_(std::move(events))
    {
        info_.device = "/dev/scripted";
        info_.actual_format = "YUYV";
        info_.actual_width = 2;
        info_.actual_height = 2;
        info_.actual_fps = 30.0;
        info_.bytes_per_line = 4;
    }

    bool nextFrame(std::vector<std::uint8_t>& payload,
                   std::string& error) override
    {
        error_kind_ = edge_ai::CameraErrorKind::None;
        if (index_ >= events_.size()) {
            error = "scripted camera ended";
            error_kind_ = edge_ai::CameraErrorKind::DeviceLost;
            return false;
        }
        const Event event = events_[index_++];
        if (event == Event::Frame) {
            payload = {0, 128, 255, 128, 0, 128, 255, 128};
            return true;
        }
        if (event == Event::Bad) {
            error = "camera frame marked with V4L2_BUF_FLAG_ERROR";
            error_kind_ = edge_ai::CameraErrorKind::BadFrame;
        } else {
            error = "camera device disconnected";
            error_kind_ = edge_ai::CameraErrorKind::DeviceLost;
        }
        return false;
    }

    const edge_ai::CameraInfo& info() const override { return info_; }
    edge_ai::CameraErrorKind lastErrorKind() const override
    {
        return error_kind_;
    }
    void close() noexcept override {}

private:
    std::vector<Event> events_;
    std::size_t index_ = 0;
    edge_ai::CameraInfo info_;
    edge_ai::CameraErrorKind error_kind_ = edge_ai::CameraErrorKind::None;
};

edge_ai::CameraOptions reconnectCameraOptions()
{
    edge_ai::CameraOptions options;
    options.device = "/dev/scripted";
    options.format = "YUYV";
    options.width = 2;
    options.height = 2;
    options.fps = 30;
    options.timeout_ms = 100;
    return options;
}

bool testCameraReconnectSuccess(std::string& error)
{
    using Event = ScriptedCameraBackend::Event;
    std::vector<std::vector<Event>> scripts = {
        {Event::Frame, Event::Lost}, {Event::Frame}};
    int factory_calls = 0;
    const edge_ai::CameraFactory factory =
        [&](const edge_ai::CameraOptions& options,
            std::string& open_error) -> std::unique_ptr<edge_ai::CameraSource> {
        if (factory_calls >= static_cast<int>(scripts.size())) {
            open_error = "no more simulated devices";
            return nullptr;
        }
        auto backend = std::make_unique<ScriptedCameraBackend>(
            scripts[static_cast<std::size_t>(factory_calls++)]);
        return std::make_unique<edge_ai::CameraSource>(
            options, std::move(backend), open_error);
    };
    std::string source_error;
    edge_ai::ReconnectingCameraSource source(
        reconnectCameraOptions(), 2, 0, nullptr, factory, source_error);
    if (!expect(source_error.empty(),
                "reconnect source construction failed: " + source_error,
                error)) {
        return false;
    }

    edge_ai::Frame first;
    edge_ai::Frame second;
    return expect(source.next(first, source_error) ==
                      edge_ai::SourceStatus::Frame &&
                      first.index == 0,
                  "first pre-reconnect frame failed", error) &&
           expect(source.next(second, source_error) ==
                      edge_ai::SourceStatus::Frame &&
                      second.index == 1,
                  "frame after reconnect failed: " + source_error, error) &&
           expect(source.reconnectCount() == 1,
                  "reconnect count is not 1", error);
}

bool testCameraReconnectExhausted(std::string& error)
{
    using Event = ScriptedCameraBackend::Event;
    int factory_calls = 0;
    const edge_ai::CameraFactory factory =
        [&](const edge_ai::CameraOptions& options,
            std::string& open_error) -> std::unique_ptr<edge_ai::CameraSource> {
        ++factory_calls;
        if (factory_calls == 1) {
            auto backend = std::make_unique<ScriptedCameraBackend>(
                std::vector<Event>{Event::Lost});
            return std::make_unique<edge_ai::CameraSource>(
                options, std::move(backend), open_error);
        }
        open_error = "simulated reopen failure";
        return nullptr;
    };
    std::string source_error;
    edge_ai::ReconnectingCameraSource source(
        reconnectCameraOptions(), 2, 0, nullptr, factory, source_error);
    edge_ai::Frame frame;
    if (!expect(source_error.empty(),
                "exhausted source construction failed: " + source_error,
                error)) {
        return false;
    }
    const edge_ai::SourceStatus status = source.next(frame, source_error);
    return expect(status == edge_ai::SourceStatus::Error,
                  "exhausted reconnect did not fail", error) &&
           expect(source.errorKind() == edge_ai::SourceErrorKind::Fatal,
                  "exhausted reconnect is not Fatal", error) &&
           expect(source_error.find("reconnect failed") != std::string::npos,
                  "exhausted reconnect error is unclear: " + source_error,
                  error) &&
           expect(factory_calls == 3,
                  "reconnect attempts did not use the configured budget",
                  error);
}

bool testCameraReconnectOpenError(std::string& error)
{
    using Event = ScriptedCameraBackend::Event;
    std::vector<std::string> logs;
    int factory_calls = 0;
    const edge_ai::CameraFactory factory =
        [&](const edge_ai::CameraOptions& options,
            std::string& open_error) -> std::unique_ptr<edge_ai::CameraSource> {
        ++factory_calls;
        if (factory_calls == 1) {
            auto backend = std::make_unique<ScriptedCameraBackend>(
                std::vector<Event>{Event::Lost});
            return std::make_unique<edge_ai::CameraSource>(
                options, std::move(backend), open_error);
        }
        // Mimic the production factory: the object is non-null even though
        // opening failed, and the error string is the failure signal.
        return std::make_unique<edge_ai::CameraSource>(
            options, std::unique_ptr<edge_ai::CameraBackend>{}, open_error);
    };
    std::string source_error;
    edge_ai::ReconnectingCameraSource source(
        reconnectCameraOptions(), 2, 0, nullptr, factory, source_error,
        [&logs](const std::string& message) { logs.push_back(message); });
    if (!expect(source_error.empty(),
                "open-error source construction failed: " + source_error,
                error)) {
        return false;
    }
    edge_ai::Frame frame;
    const edge_ai::SourceStatus status = source.next(frame, source_error);
    if (!expect(status == edge_ai::SourceStatus::Error,
                "non-null open failure was treated as success", error) ||
        !expect(source.errorKind() == edge_ai::SourceErrorKind::Fatal,
                "non-null open failure was not fatal", error) ||
        !expect(factory_calls == 3,
                "non-null open failure did not use the reconnect budget", error) ||
        !expect(source_error.find("reconnect failed") != std::string::npos,
                "open-failure diagnosis is unclear: " + source_error, error)) {
        return false;
    }
    if (!expect(logs.size() >= 2,
                "open-failure reconnect did not log loss/exhaustion", error) ||
        !expect(logs.front().find("camera lost") != std::string::npos,
                "first reconnect log is not a loss event", error) ||
        !expect(logs.back().find("reconnect failed") != std::string::npos,
                "last reconnect log is not exhaustion", error)) {
        return false;
    }

    // The initial factory call must reject the same non-null/error shape.
    const edge_ai::CameraFactory failing_factory =
        [](const edge_ai::CameraOptions& options,
           std::string& open_error) -> std::unique_ptr<edge_ai::CameraSource> {
        return std::make_unique<edge_ai::CameraSource>(
            options, std::unique_ptr<edge_ai::CameraBackend>{}, open_error);
    };
    std::string initial_error;
    edge_ai::ReconnectingCameraSource rejected(
        reconnectCameraOptions(), 2, 0, nullptr, failing_factory, initial_error);
    return expect(!initial_error.empty(),
                  "initial non-null open failure was accepted", error);
}

bool testCameraReconnectNoFrameBudget(std::string& error)
{
    class NoFrameBackend final : public edge_ai::CameraBackend {
    public:
        NoFrameBackend()
        {
            info_.device = "/dev/no-frame";
            info_.actual_format = "YUYV";
            info_.actual_width = 2;
            info_.actual_height = 2;
            info_.actual_fps = 30.0;
            info_.bytes_per_line = 4;
        }
        bool nextFrame(std::vector<std::uint8_t>&,
                       std::string& read_error) override
        {
            read_error = "camera read timeout after 100 ms";
            return false;
        }
        const edge_ai::CameraInfo& info() const override { return info_; }
        edge_ai::CameraErrorKind lastErrorKind() const override
        {
            return edge_ai::CameraErrorKind::Timeout;
        }
        void close() noexcept override {}

    private:
        edge_ai::CameraInfo info_;
    };

    std::vector<std::string> logs;
    int factory_calls = 0;
    const edge_ai::CameraFactory factory =
        [&](const edge_ai::CameraOptions& options,
            std::string& open_error) -> std::unique_ptr<edge_ai::CameraSource> {
        ++factory_calls;
        // Every reopen succeeds, but no frame ever arrives.
        return std::make_unique<edge_ai::CameraSource>(
            options, std::make_unique<NoFrameBackend>(), open_error);
    };
    std::string source_error;
    edge_ai::ReconnectingCameraSource source(
        reconnectCameraOptions(), 3, 0, nullptr, factory, source_error,
        [&logs](const std::string& message) { logs.push_back(message); });
    if (!expect(source_error.empty(),
                "no-frame source construction failed: " + source_error,
                error)) {
        return false;
    }
    edge_ai::Frame frame;
    const edge_ai::SourceStatus status = source.next(frame, source_error);
    if (!expect(status == edge_ai::SourceStatus::Error,
                "no-frame outage did not become fatal", error) ||
        !expect(source.errorKind() == edge_ai::SourceErrorKind::Fatal,
                "no-frame outage is not Fatal", error) ||
        !expect(factory_calls == 4,
                "no-frame outage ignored the reconnect budget: " +
                    std::to_string(factory_calls) + " opens",
                error) ||
        !expect(source.reconnectCount() == 3,
                "no-frame outage did not count three successful reopens",
                error) ||
        !expect(source_error.find("no valid frame") != std::string::npos,
                "no-frame exhaustion is not diagnostic: " + source_error,
                error)) {
        return false;
    }
    if (!expect(logs.size() >= 5,
                "no-frame outage did not log loss/attempts/exhaustion", error) ||
        !expect(logs.front().find("camera lost") != std::string::npos,
                "no-frame outage did not log the loss", error) ||
        !expect(logs.back().find("reconnect failed") != std::string::npos,
                "no-frame outage did not log exhaustion", error)) {
        return false;
    }
    return true;
}

bool testCameraReconnectStop(std::string& error)
{
    using Event = ScriptedCameraBackend::Event;
    std::atomic<bool> stop{false};
    int factory_calls = 0;
    const edge_ai::CameraFactory factory =
        [&](const edge_ai::CameraOptions& options,
            std::string& open_error) -> std::unique_ptr<edge_ai::CameraSource> {
        ++factory_calls;
        auto backend = std::make_unique<ScriptedCameraBackend>(
            std::vector<Event>{Event::Lost});
        return std::make_unique<edge_ai::CameraSource>(
            options, std::move(backend), open_error);
    };
    std::string source_error;
    edge_ai::ReconnectingCameraSource source(
        reconnectCameraOptions(), 3, 400, &stop, factory, source_error);
    if (!expect(source_error.empty(),
                "stop source construction failed: " + source_error, error)) {
        return false;
    }

    edge_ai::SourceStatus status = edge_ai::SourceStatus::Error;
    std::thread worker([&] {
        edge_ai::Frame frame;
        status = source.next(frame, source_error);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stop.store(true);
    worker.join();
    return expect(status == edge_ai::SourceStatus::End,
                  "stop during reconnect did not return End", error) &&
           expect(factory_calls == 1,
                  "stop during reconnect still attempted to reopen", error);
}

bool testCameraBadFrameSequence(std::string& error)
{
    using Event = ScriptedCameraBackend::Event;
    std::vector<std::vector<Event>> scripts = {{Event::Bad, Event::Frame}};
    int factory_calls = 0;
    const edge_ai::CameraFactory factory =
        [&](const edge_ai::CameraOptions& options,
            std::string& open_error) -> std::unique_ptr<edge_ai::CameraSource> {
        auto backend = std::make_unique<ScriptedCameraBackend>(
            scripts[static_cast<std::size_t>(factory_calls++)]);
        return std::make_unique<edge_ai::CameraSource>(
            options, std::move(backend), open_error);
    };
    std::string source_error;
    edge_ai::ReconnectingCameraSource source(
        reconnectCameraOptions(), 2, 0, nullptr, factory, source_error);
    if (!expect(source_error.empty(),
                "bad-frame source construction failed: " + source_error,
                error)) {
        return false;
    }
    edge_ai::Frame frame;
    if (!expect(source.next(frame, source_error) ==
                    edge_ai::SourceStatus::Error,
                "bad frame was not reported", error) ||
        !expect(source.errorKind() == edge_ai::SourceErrorKind::DataError,
                "bad frame was classified as fatal", error) ||
        !expect(frame.index == 0, "bad frame did not consume index 0", error)) {
        return false;
    }
    return expect(source.next(frame, source_error) ==
                      edge_ai::SourceStatus::Frame &&
                      frame.index == 1,
                  "frame after bad frame did not use index 1", error);
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
    edge_ai::CameraErrorKind lastErrorKind() const override
    {
        return edge_ai::CameraErrorKind::Timeout;
    }
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
        {"camera_buffer_validation", testCameraBufferValidation},
        {"camera_busy_error", testCameraBusyError},
        {"camera_read_deadline", testCameraReadDeadline},
        {"camera_read_stop", testCameraReadStop},
        {"camera_source_external_stop", testCameraSourceExternalStop},
        {"camera_colorimetry", testCameraColorimetry},
        {"camera_reconnect_success", testCameraReconnectSuccess},
        {"camera_reconnect_exhausted", testCameraReconnectExhausted},
        {"camera_reconnect_open_error", testCameraReconnectOpenError},
        {"camera_reconnect_no_frame_budget", testCameraReconnectNoFrameBudget},
        {"camera_reconnect_stop", testCameraReconnectStop},
        {"camera_bad_frame_sequence", testCameraBadFrameSequence},
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

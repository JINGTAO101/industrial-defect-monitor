#include <cmath>
#include <cstddef>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "image_preprocess.hpp"
#include "model_contract.hpp"
#include "postprocess.hpp"

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
        {"postprocess_shape_mismatch", testPostprocessShapeMismatch},
        {"class_mapping_unknown", testClassMappingUnknown},
        {"contract_validation", testContractValidation},
        {"class_metadata_parser", testClassMetadataParser},
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

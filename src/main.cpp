#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "model_runner.hpp"
#include "postprocess.hpp"
#include "system_info.hpp"

#ifndef EDGE_AI_VERSION
#define EDGE_AI_VERSION "0.2.0"
#endif

namespace {

constexpr std::string_view kProgram = "edge_ai";

struct Options {
    bool show_help = false;
    bool show_version = false;
    std::string model_path;
    std::string image_path;
    std::string json_out;
    std::string dump_input;
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
};

void printUsage(std::ostream& os)
{
    os << "edge_ai - Industrial defect monitoring system (edge AI)\n"
       << "\n"
       << "Usage:\n"
       << "  edge_ai [OPTION]\n"
       << "  edge_ai --model <model.onnx> --image <image> [OPTIONS]\n"
       << "\n"
       << "Options:\n"
       << "  -h, --help           Show this help message and exit\n"
       << "  -V, --version        Show version information and exit\n"
       << "  --model <path>       ONNX model path (required for inference)\n"
       << "  --image <path>       Input image path (required for inference)\n"
       << "  --conf <float>       Confidence threshold in (0, 1] (default 0.25)\n"
       << "  --iou <float>        NMS IoU threshold in (0, 1] (default 0.45)\n"
       << "  --json-out <path>    Write machine-readable detection result JSON\n"
       << "  --dump-input <path>  Write the float32 NCHW input tensor (raw)\n"
       << "\n"
       << "With no --model/--image, edge_ai prints its startup banner.\n"
       << "Unknown options are rejected with a non-zero exit code.\n";
}

int printError(const std::string& message)
{
    std::cerr << kProgram << ": " << message << "\n"
              << "Try '" << kProgram << " --help' for more information.\n";
    return 2;
}

bool parseFloatOption(
    const std::string& name,
    const std::string& value,
    float& output,
    std::string& error)
{
    try {
        std::size_t consumed = 0;
        const float parsed = std::stof(value, &consumed);
        if (consumed != value.size()) {
            error = "invalid value for " + name + ": '" + value + "'";
            return false;
        }
        if (!(parsed > 0.0f && parsed <= 1.0f)) {
            error = name + " must be in (0, 1], got " + value;
            return false;
        }
        output = parsed;
        return true;
    } catch (const std::exception&) {
        error = "invalid value for " + name + ": '" + value + "'";
        return false;
    }
}

bool parseOptions(int argc, char* argv[], Options& options, std::string& error)
{
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];

        if (argument == "-h" || argument == "--help") {
            options.show_help = true;
            continue;
        }
        if (argument == "-V" || argument == "--version") {
            options.show_version = true;
            continue;
        }

        const auto take_value = [&](const char* name, std::string& target) {
            if (i + 1 >= argc) {
                error = std::string("missing value for ") + name;
                return false;
            }
            target = argv[++i];
            if (target.empty()) {
                error = std::string("empty value for ") + name;
                return false;
            }
            return true;
        };

        if (argument == "--model") {
            if (!take_value("--model", options.model_path)) {
                return false;
            }
        } else if (argument == "--image") {
            if (!take_value("--image", options.image_path)) {
                return false;
            }
        } else if (argument == "--json-out") {
            if (!take_value("--json-out", options.json_out)) {
                return false;
            }
        } else if (argument == "--dump-input") {
            if (!take_value("--dump-input", options.dump_input)) {
                return false;
            }
        } else if (argument == "--conf") {
            std::string value;
            if (!take_value("--conf", value) ||
                !parseFloatOption("--conf", value,
                                  options.conf_threshold, error)) {
                return false;
            }
        } else if (argument == "--iou") {
            std::string value;
            if (!take_value("--iou", value) ||
                !parseFloatOption("--iou", value,
                                  options.iou_threshold, error)) {
                return false;
            }
        } else if (!argument.empty() && argument.front() == '-') {
            error = "unknown option '" + std::string(argument) + "'";
            return false;
        } else {
            error = "unexpected argument '" + std::string(argument) + "'";
            return false;
        }
    }

    if (options.show_help || options.show_version) {
        return true;
    }
    if (options.model_path.empty() != options.image_path.empty()) {
        error = "both --model and --image are required for inference";
        return false;
    }
    if (options.model_path.empty()) {
        if (!options.json_out.empty() || !options.dump_input.empty()) {
            error = "--json-out/--dump-input require --model and --image";
            return false;
        }
    }
    return true;
}

std::string jsonEscape(const std::string& input)
{
    std::ostringstream os;
    for (const unsigned char c : input) {
        switch (c) {
            case '"':
                os << "\\\"";
                break;
            case '\\':
                os << "\\\\";
                break;
            case '\b':
                os << "\\b";
                break;
            case '\f':
                os << "\\f";
                break;
            case '\n':
                os << "\\n";
                break;
            case '\r':
                os << "\\r";
                break;
            case '\t':
                os << "\\t";
                break;
            default:
                if (c < 0x20) {
                    os << "\\u" << std::hex << std::setw(4)
                       << std::setfill('0') << static_cast<int>(c) << std::dec;
                } else {
                    os << static_cast<char>(c);
                }
        }
    }
    return os.str();
}

void writeShape(std::ostream& os, const std::vector<int64_t>& shape)
{
    os << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        os << shape[i];
    }
    os << "]";
}

bool ensureParentDirectory(const std::string& path, std::string& error)
{
    const std::filesystem::path parent =
        std::filesystem::path(path).parent_path();
    if (parent.empty()) {
        return true;
    }
    std::error_code fs_error;
    std::filesystem::create_directories(parent, fs_error);
    if (fs_error) {
        error = "cannot create directory '" + parent.string() + "': " +
                fs_error.message();
        return false;
    }
    return true;
}

bool writeJsonResult(
    const std::string& path,
    const edge_ai::ModelInfo& info,
    const edge_ai::PreprocessedInput& input,
    float min_value,
    float max_value,
    const edge_ai::PostprocessOptions& options,
    const edge_ai::PostprocessResult& result,
    const std::string& image_path,
    std::string& error)
{
    if (!ensureParentDirectory(path, error)) {
        return false;
    }

    std::ostringstream os;
    os << std::setprecision(9);
    os << "{\n";
    os << "  \"model\": {\n";
    os << "    \"path\": \"" << jsonEscape(info.path) << "\",\n";
    os << "    \"sha256\": \"" << jsonEscape(info.sha256) << "\",\n";
    os << "    \"ort_version\": \"" << jsonEscape(info.ort_version) << "\",\n";
    os << "    \"providers\": [";
    for (std::size_t i = 0; i < info.providers.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        os << "\"" << jsonEscape(info.providers[i]) << "\"";
    }
    os << "]\n";
    os << "  },\n";
    os << "  \"input_contract\": {\n";
    os << "    \"name\": \"" << jsonEscape(info.input_name) << "\",\n";
    os << "    \"shape\": ";
    writeShape(os, info.input_shape);
    os << ",\n";
    os << "    \"type\": \"" << jsonEscape(info.input_type) << "\",\n";
    os << "    \"layout\": \"NCHW\",\n";
    os << "    \"color_order\": \"BGR\",\n";
    os << "    \"normalization\": \"divide by 255 only; no mean/std\"\n";
    os << "  },\n";
    os << "  \"output_contract\": {\n";
    os << "    \"name\": \"" << jsonEscape(info.output_name) << "\",\n";
    os << "    \"shape\": ";
    writeShape(os, info.output_shape);
    os << ",\n";
    os << "    \"type\": \"" << jsonEscape(info.output_type) << "\",\n";
    os << "    \"layout\": \"channel-first [1, 4+nc, num_preds]\",\n";
    os << "    \"nms_included\": false\n";
    os << "  },\n";
    os << "  \"class_names\": {";
    bool first = true;
    for (const auto& entry : info.class_names) {
        if (!first) {
            os << ", ";
        }
        first = false;
        os << "\"" << entry.first << "\": \""
           << jsonEscape(entry.second) << "\"";
    }
    os << "},\n";
    os << "  \"thresholds\": {\"conf\": " << std::setprecision(6)
       << options.conf_threshold << ", \"iou\": " << options.iou_threshold
       << "},\n";
    os << std::setprecision(9);
    os << "  \"image\": {\n";
    os << "    \"path\": \"" << jsonEscape(image_path) << "\",\n";
    os << "    \"width\": " << input.letterbox.original_width << ",\n";
    os << "    \"height\": " << input.letterbox.original_height << ",\n";
    os << "    \"letterbox\": {\n";
    os << "      \"ratio\": " << input.letterbox.ratio << ",\n";
    os << "      \"pad_left\": " << input.letterbox.pad_left << ",\n";
    os << "      \"pad_top\": " << input.letterbox.pad_top << "\n";
    os << "    }\n";
    os << "  },\n";
    os << "  \"input_tensor\": {\"shape\": [1, 3, 640, 640], "
          "\"type\": \"float32\", \"min\": ";
    os << min_value << ", \"max\": " << max_value << "},\n";
    os << "  \"detections\": [";
    for (std::size_t i = 0; i < result.detections.size(); ++i) {
        const auto& detection = result.detections[i];
        if (i != 0) {
            os << ",";
        }
        os << "\n    {\"class_id\": " << detection.class_id
           << ", \"class_name\": \"" << jsonEscape(detection.class_name)
           << "\", \"confidence\": " << detection.confidence
           << ", \"bbox_xyxy\": [" << detection.x1 << ", " << detection.y1
           << ", " << detection.x2 << ", " << detection.y2 << "]}";
    }
    if (!result.detections.empty()) {
        os << "\n  ";
    }
    os << "],\n";
    os << "  \"anomalies\": [";
    for (std::size_t i = 0; i < result.anomalies.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        os << "\"" << jsonEscape(result.anomalies[i]) << "\"";
    }
    os << "]\n";
    os << "}\n";

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        error = "cannot open JSON output: " + path;
        return false;
    }
    output << os.str();
    if (!output) {
        error = "failed to write JSON output: " + path;
        return false;
    }
    return true;
}

bool writeInputTensor(
    const std::string& path,
    const edge_ai::PreprocessedInput& input,
    std::string& error)
{
    if (!ensureParentDirectory(path, error)) {
        return false;
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        error = "cannot open input tensor output: " + path;
        return false;
    }
    output.write(
        reinterpret_cast<const char*>(input.data.data()),
        static_cast<std::streamsize>(input.data.size() * sizeof(float)));
    if (!output) {
        error = "failed to write input tensor output: " + path;
        return false;
    }
    return true;
}

int runInference(const Options& options)
{
    std::string error;
    const auto runner = edge_ai::ModelRunner::load(options.model_path, error);
    if (!runner) {
        std::cerr << kProgram << ": " << error << "\n";
        return 1;
    }

    cv::Mat image = cv::imread(options.image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << kProgram << ": failed to read image: "
                  << options.image_path << "\n";
        return 1;
    }

    edge_ai::PreprocessedInput input;
    try {
        input = runner->preprocess(image);
    } catch (const std::exception& exception) {
        std::cerr << kProgram << ": preprocessing failed: "
                  << exception.what() << "\n";
        return 1;
    }

    const auto [min_it, max_it] =
        std::minmax_element(input.data.begin(), input.data.end());
    const float min_value = *min_it;
    const float max_value = *max_it;

    if (!options.dump_input.empty() &&
        !writeInputTensor(options.dump_input, input, error)) {
        std::cerr << kProgram << ": " << error << "\n";
        return 1;
    }

    const edge_ai::PostprocessOptions postprocess_options{
        options.conf_threshold, options.iou_threshold};
    const edge_ai::PostprocessResult result =
        runner->infer(input, postprocess_options);
    if (!result.error.empty()) {
        std::cerr << kProgram << ": " << result.error << "\n";
        return 1;
    }

    const edge_ai::ModelInfo& info = runner->info();
    std::cout << "ONNX Runtime " << info.ort_version
              << " providers=CPUExecutionProvider\n";
    std::cout << "model: " << info.path << "\n";
    std::cout << "model sha256: " << info.sha256 << "\n";
    std::cout << "input:  " << info.input_name << " ";
    writeShape(std::cout, info.input_shape);
    std::cout << " " << info.input_type << "\n";
    std::cout << "output: " << info.output_name << " ";
    writeShape(std::cout, info.output_shape);
    std::cout << " " << info.output_type << "\n";
    std::cout << "classes: ";
    bool first_class = true;
    for (const auto& entry : info.class_names) {
        if (!first_class) {
            std::cout << ", ";
        }
        first_class = false;
        std::cout << entry.first << "=" << entry.second;
    }
    std::cout << "\n";
    std::cout << "image: " << options.image_path << " "
              << image.cols << "x" << image.rows << "\n";
    std::cout << "preprocess: letterbox ratio="
              << input.letterbox.ratio
              << " pad=(" << input.letterbox.pad_left << ","
              << input.letterbox.pad_top << ") BGR /255 NCHW\n";
    std::cout << "input tensor: [1, 3, 640, 640] float32 min="
              << min_value << " max=" << max_value << "\n";
    std::cout << "thresholds: conf>" << options.conf_threshold
              << " iou=" << options.iou_threshold << "\n";
    std::cout << "detections: " << result.detections.size() << "\n";
    for (const auto& detection : result.detections) {
        std::cout << "  class_id=" << detection.class_id << " "
                  << detection.class_name
                  << " conf=" << std::fixed << std::setprecision(6)
                  << detection.confidence << std::defaultfloat
                  << " bbox_xyxy=[" << detection.x1 << ", "
                  << detection.y1 << ", " << detection.x2 << ", "
                  << detection.y2 << "]\n";
    }
    for (const auto& anomaly : result.anomalies) {
        std::cerr << "warning: " << anomaly << "\n";
    }

    if (!options.json_out.empty() &&
        !writeJsonResult(options.json_out, info, input, min_value, max_value,
                         postprocess_options, result, options.image_path,
                         error)) {
        std::cerr << kProgram << ": " << error << "\n";
        return 1;
    }

    return 0;
}

}  // namespace

int main(int argc, char* argv[])
{
    Options options;
    std::string error;
    if (!parseOptions(argc, argv, options, error)) {
        return printError(error);
    }

    if (options.show_help) {
        printUsage(std::cout);
        return 0;
    }
    if (options.show_version) {
        std::cout << kProgram << " " << EDGE_AI_VERSION << "\n";
        return 0;
    }
    if (options.model_path.empty()) {
        std::cout << "====================================\n";
        std::cout << " Industrial Defect Monitor System\n";
        std::cout << "====================================\n";
        std::cout << getSystemMessage() << "\n";
        return 0;
    }

    return runInference(options);
}

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

#include "batch_processor.hpp"
#include "image_source.hpp"
#include "model_runner.hpp"
#include "postprocess.hpp"
#include "quality.hpp"
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
    std::string image_dir;
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
       << "  edge_ai --model <model.onnx> (--image <image> | --image-dir <dir>) [OPTIONS]\n"
       << "\n"
       << "Options:\n"
       << "  -h, --help           Show this help message and exit\n"
       << "  -V, --version        Show version information and exit\n"
       << "  --model <path>       ONNX model path (required for inference)\n"
       << "  --image <path>       Single input image\n"
       << "  --image-dir <dir>    Batch input directory\n"
       << "  --conf <float>       Confidence threshold in (0, 1] (default 0.25)\n"
       << "  --iou <float>        NMS IoU threshold in (0, 1] (default 0.45)\n"
       << "  --json-out <path>    Write detection JSON (single result or batch summary)\n"
       << "  --dump-input <path>  Write the float32 NCHW input tensor (single image only)\n"
       << "\n"
       << "Supported image extensions: jpg, jpeg, png, bmp, tif, tiff.\n"
       << "Batch processing continues after corrupt images, reports per-file errors,\n"
       << "and exits 1 if any image failed. Quality rule: NG if at least one\n"
       << "detection has confidence > --conf; otherwise OK.\n"
       << "Values beginning with '-' must use the --option=value form\n"
       << "(for example: --model=-model.onnx).\n"
       << "With no --model/image input, edge_ai prints its startup banner.\n"
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

std::filesystem::path normalizedPath(const std::string& path)
{
    std::error_code error_code;
    std::filesystem::path absolute =
        std::filesystem::absolute(path, error_code);
    if (error_code) {
        absolute = std::filesystem::path(path);
        error_code.clear();
    }

    // Resolve the final symlink chain explicitly. weakly_canonical() leaves a
    // dangling final symlink as the link path, so "alias -> missing.json"
    // would otherwise compare differently from "missing.json" even though
    // writing through the alias creates/overwrites that target.
    for (int depth = 0; depth < 40; ++depth) {
        error_code.clear();
        const std::filesystem::file_status status =
            std::filesystem::symlink_status(absolute, error_code);
        if (error_code ||
            status.type() != std::filesystem::file_type::symlink) {
            break;
        }
        error_code.clear();
        std::filesystem::path target =
            std::filesystem::read_symlink(absolute, error_code);
        if (error_code || target.empty()) {
            break;
        }
        if (target.is_relative()) {
            target = absolute.parent_path() / target;
        }
        absolute = target.lexically_normal();
    }

    error_code.clear();
    const std::filesystem::path canonical =
        std::filesystem::weakly_canonical(absolute, error_code);
    if (!error_code) {
        return canonical;
    }
    return absolute.lexically_normal();
}

// Detects identical paths, relative aliases, symlinks and hard links.
bool pathsConflict(const std::string& first, const std::string& second)
{
    if (first.empty() || second.empty()) {
        return false;
    }

    std::error_code error_code;
    const bool first_exists = std::filesystem::exists(first, error_code);
    error_code.clear();
    const bool second_exists = std::filesystem::exists(second, error_code);
    if (first_exists && second_exists) {
        error_code.clear();
        if (std::filesystem::equivalent(first, second, error_code) &&
            !error_code) {
            return true;
        }
    }
    return normalizedPath(first) == normalizedPath(second);
}

bool validateOutputPaths(
    const Options& options,
    const std::vector<std::string>& input_images,
    std::string& error)
{
    if (!options.dump_input.empty() && !options.json_out.empty() &&
        pathsConflict(options.dump_input, options.json_out)) {
        error = "output paths '" + options.dump_input + "' and '" +
                options.json_out + "' refer to the same file";
        return false;
    }

    const auto check_output = [&](const std::string& output) {
        if (output.empty()) {
            return true;
        }
        if (pathsConflict(output, options.model_path)) {
            error = "output path '" + output + "' conflicts with model path '" +
                    options.model_path + "'";
            return false;
        }
        for (const std::string& input : input_images) {
            if (pathsConflict(output, input)) {
                error = "output path '" + output +
                        "' conflicts with input image path '" + input + "'";
                return false;
            }
        }
        return true;
    };

    return check_output(options.dump_input) && check_output(options.json_out);
}

bool parseOptions(int argc, char* argv[], Options& options, std::string& error)
{
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        std::string option = argument;
        std::string inline_value;
        bool has_inline_value = false;
        if (argument.rfind("--", 0) == 0) {
            const std::size_t equals = argument.find('=');
            if (equals != std::string::npos) {
                option = argument.substr(0, equals);
                inline_value = argument.substr(equals + 1);
                has_inline_value = true;
            }
        }

        if (option == "-h" || option == "--help") {
            if (has_inline_value) {
                error = option + " does not take a value";
                return false;
            }
            options.show_help = true;
            continue;
        }
        if (option == "-V" || option == "--version") {
            if (has_inline_value) {
                error = option + " does not take a value";
                return false;
            }
            options.show_version = true;
            continue;
        }

        const auto take_value = [&](const char* name, std::string& target) {
            if (has_inline_value) {
                if (inline_value.empty()) {
                    error = std::string("empty value for ") + name;
                    return false;
                }
                target = inline_value;
                return true;
            }
            if (i + 1 >= argc) {
                error = std::string("missing value for ") + name;
                return false;
            }
            const std::string_view next = argv[i + 1];
            if (next.empty() || next.front() == '-') {
                error = std::string("missing value for ") + name +
                        " (use " + name +
                        "=<value> when the value starts with '-')";
                return false;
            }
            target = std::string(next);
            ++i;
            return true;
        };

        if (option == "--model") {
            if (!take_value("--model", options.model_path)) {
                return false;
            }
        } else if (option == "--image") {
            if (!take_value("--image", options.image_path)) {
                return false;
            }
        } else if (option == "--image-dir") {
            if (!take_value("--image-dir", options.image_dir)) {
                return false;
            }
        } else if (option == "--json-out") {
            if (!take_value("--json-out", options.json_out)) {
                return false;
            }
        } else if (option == "--dump-input") {
            if (!take_value("--dump-input", options.dump_input)) {
                return false;
            }
        } else if (option == "--conf") {
            std::string value;
            if (!take_value("--conf", value) ||
                !parseFloatOption("--conf", value,
                                  options.conf_threshold, error)) {
                return false;
            }
        } else if (option == "--iou") {
            std::string value;
            if (!take_value("--iou", value) ||
                !parseFloatOption("--iou", value,
                                  options.iou_threshold, error)) {
                return false;
            }
        } else if (!option.empty() && option.front() == '-') {
            error = "unknown option '" + option + "'";
            return false;
        } else {
            error = "unexpected argument '" + argument + "'";
            return false;
        }
    }

    if (options.show_help || options.show_version) {
        return true;
    }

    const bool has_single_image = !options.image_path.empty();
    const bool has_image_dir = !options.image_dir.empty();
    if (has_single_image && has_image_dir) {
        error = "--image and --image-dir are mutually exclusive";
        return false;
    }
    if (options.model_path.empty()) {
        if (has_single_image || has_image_dir) {
            error = "--model is required for inference";
            return false;
        }
        if (!options.json_out.empty() || !options.dump_input.empty()) {
            error = "--json-out/--dump-input require --model and an image input";
            return false;
        }
        return true;
    }
    if (!has_single_image && !has_image_dir) {
        error = "one of --image or --image-dir is required";
        return false;
    }
    if (has_image_dir && !options.dump_input.empty()) {
        error = "--dump-input is only supported with --image";
        return false;
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
    const edge_ai::QualityResult& quality,
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
    os << "  \"quality\": {\"status\": \""
       << edge_ai::qualityLabel(quality.is_ng) << "\", \"is_ng\": "
       << (quality.is_ng ? "true" : "false")
       << ", \"confidence_threshold\": " << quality.confidence_threshold
       << ", \"total_detections\": " << quality.total_detections
       << ", \"qualifying_defects\": " << quality.defects.size() << "},\n";
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

void printModelInfo(const edge_ai::ModelInfo& info)
{
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
}

bool writeBatchJson(
    const std::string& path,
    const edge_ai::ModelInfo& info,
    const edge_ai::PostprocessOptions& options,
    const edge_ai::BatchResult& batch,
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
    os << "  \"thresholds\": {\"conf\": " << std::setprecision(6)
       << options.conf_threshold << ", \"iou\": " << options.iou_threshold
       << "},\n";
    os << std::setprecision(9);
    os << "  \"quality_rule\": \"NG if at least one detection has confidence > threshold\",\n";
    os << "  \"frames\": [";
    for (std::size_t i = 0; i < batch.frames.size(); ++i) {
        const edge_ai::FrameResult& frame = batch.frames[i];
        const char* status =
            frame.error.empty() ? (frame.quality.is_ng ? "NG" : "OK") : "ERROR";
        if (i != 0) {
            os << ",";
        }
        os << "\n    {\n";
        os << "      \"source\": \"" << jsonEscape(frame.source) << "\",\n";
        os << "      \"status\": \"" << status << "\",\n";
        os << "      \"error\": \"" << jsonEscape(frame.error) << "\",\n";
        os << "      \"width\": " << frame.width << ",\n";
        os << "      \"height\": " << frame.height << ",\n";
        os << "      \"letterbox\": {\"ratio\": " << frame.letterbox.ratio
           << ", \"pad_left\": " << frame.letterbox.pad_left
           << ", \"pad_top\": " << frame.letterbox.pad_top << "},\n";
        os << "      \"detections\": [";
        for (std::size_t j = 0; j < frame.detections.size(); ++j) {
            const auto& detection = frame.detections[j];
            if (j != 0) {
                os << ", ";
            }
            os << "{\"class_id\": " << detection.class_id
               << ", \"class_name\": \"" << jsonEscape(detection.class_name)
               << "\", \"confidence\": " << detection.confidence
               << ", \"bbox_xyxy\": [" << detection.x1 << ", "
               << detection.y1 << ", " << detection.x2 << ", "
               << detection.y2 << "]}";
        }
        os << "],\n";
        os << "      \"anomalies\": [";
        for (std::size_t j = 0; j < frame.anomalies.size(); ++j) {
            if (j != 0) {
                os << ", ";
            }
            os << "\"" << jsonEscape(frame.anomalies[j]) << "\"";
        }
        os << "],\n";
        os << "      \"quality\": {\"status\": \"" << status
           << "\", \"is_ng\": " << (frame.quality.is_ng ? "true" : "false")
           << ", \"confidence_threshold\": " << options.conf_threshold
           << ", \"total_detections\": " << frame.quality.total_detections
           << ", \"qualifying_defects\": " << frame.quality.defects.size()
           << "}\n";
        os << "    }";
    }
    if (!batch.frames.empty()) {
        os << "\n  ";
    }
    os << "],\n";
    os << "  \"summary\": {\"total\": " << batch.frames.size()
       << ", \"ok\": " << batch.ok_count << ", \"ng\": " << batch.ng_count
       << ", \"errors\": " << batch.error_count
       << ", \"skipped_unsupported\": " << batch.skipped_unsupported << "}\n";
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

int runSingleImage(const Options& options)
{
    std::string error;
    edge_ai::SingleImageSource source(options.image_path, error);
    if (!error.empty()) {
        std::cerr << kProgram << ": " << error << "\n";
        return 1;
    }

    const auto runner = edge_ai::ModelRunner::load(options.model_path, error);
    if (!runner) {
        std::cerr << kProgram << ": " << error << "\n";
        return 1;
    }

    edge_ai::Frame frame;
    std::string source_error;
    const edge_ai::SourceStatus status = source.next(frame, source_error);
    if (status != edge_ai::SourceStatus::Frame) {
        std::cerr << kProgram << ": "
                  << (source_error.empty() ? "no image available"
                                           : source_error)
                  << "\n";
        return 1;
    }

    edge_ai::PreprocessedInput input;
    try {
        input = runner->preprocess(frame.image);
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

    edge_ai::QualityResult quality;
    std::string quality_error;
    if (!edge_ai::evaluateQuality(result.detections, options.conf_threshold,
                                  quality, quality_error)) {
        std::cerr << kProgram << ": " << quality_error << "\n";
        return 1;
    }

    const edge_ai::ModelInfo& info = runner->info();
    printModelInfo(info);
    std::cout << "image: " << frame.source << " "
              << frame.image.cols << "x" << frame.image.rows << "\n";
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
    std::cout << "quality: " << edge_ai::qualityLabel(quality.is_ng)
              << " (confidence>" << options.conf_threshold
              << ", qualifying=" << quality.defects.size() << "/"
              << quality.total_detections << ")\n";
    for (const auto& anomaly : result.anomalies) {
        std::cerr << "warning: " << anomaly << "\n";
    }

    if (!options.json_out.empty() &&
        !writeJsonResult(options.json_out, info, input, min_value, max_value,
                         postprocess_options, result, quality, frame.source,
                         error)) {
        std::cerr << kProgram << ": " << error << "\n";
        return 1;
    }

    return 0;
}

int runBatch(const Options& options)
{
    std::string error;
    edge_ai::DirectoryImageSource source(options.image_dir, error);
    if (!error.empty()) {
        std::cerr << kProgram << ": " << error << "\n";
        return 1;
    }

    const auto runner = edge_ai::ModelRunner::load(options.model_path, error);
    if (!runner) {
        std::cerr << kProgram << ": " << error << "\n";
        return 1;
    }

    const edge_ai::PostprocessOptions postprocess_options{
        options.conf_threshold, options.iou_threshold};
    edge_ai::BatchResult batch;
    std::string fatal_error;
    if (!edge_ai::processFrames(source, *runner, postprocess_options, batch,
                                fatal_error)) {
        std::cerr << kProgram << ": " << fatal_error << "\n";
        return 1;
    }

    printModelInfo(runner->info());
    std::cout << "batch input: " << options.image_dir << "\n";
    std::cout << "quality rule: NG if at least one detection has confidence > "
              << options.conf_threshold << "\n";
    for (const auto& frame : batch.frames) {
        if (!frame.error.empty()) {
            std::cout << "[ERROR] " << frame.source << ": " << frame.error
                      << "\n";
        } else {
            std::cout << "[" << edge_ai::qualityLabel(frame.quality.is_ng)
                      << "] " << frame.source
                      << " detections=" << frame.detections.size() << "\n";
        }
        for (const auto& anomaly : frame.anomalies) {
            std::cerr << "warning: " << frame.source << ": " << anomaly
                      << "\n";
        }
    }
    std::cout << "summary: total=" << batch.frames.size()
              << " ok=" << batch.ok_count << " ng=" << batch.ng_count
              << " errors=" << batch.error_count
              << " skipped_unsupported=" << batch.skipped_unsupported << "\n";

    if (!options.json_out.empty() &&
        !writeBatchJson(options.json_out, runner->info(), postprocess_options,
                        batch, error)) {
        std::cerr << kProgram << ": " << error << "\n";
        return 1;
    }

    return batch.error_count == 0 ? 0 : 1;
}

int runInference(const Options& options)
{
    std::vector<std::string> input_images;
    if (!options.image_dir.empty()) {
        input_images = edge_ai::listSupportedImageFiles(options.image_dir);
    } else {
        input_images.push_back(options.image_path);
    }

    std::string path_error;
    if (!validateOutputPaths(options, input_images, path_error)) {
        std::cerr << kProgram << ": " << path_error << "\n";
        return 2;
    }

    if (!options.image_dir.empty()) {
        return runBatch(options);
    }
    return runSingleImage(options);
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

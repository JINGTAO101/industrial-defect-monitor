#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

#include "batch_processor.hpp"
#include "camera_source.hpp"
#include "image_source.hpp"
#include "model_runner.hpp"
#include "postprocess.hpp"
#include "quality.hpp"
#include "system_info.hpp"
#include "threaded_pipeline.hpp"

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
    std::size_t queue_capacity = 4;
    bool queue_capacity_set = false;

    std::string camera_path;
    bool capture_only = false;
    std::size_t max_frames = 30;
    std::size_t max_frames_set = false;
    std::string camera_format = "YUYV";
    int camera_width = 640;
    int camera_height = 480;
    int camera_fps = 30;
    int camera_timeout_ms = 2000;
    bool camera_options_set = false;
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
       << "  --queue-capacity <n> Batch pipeline queue size (>0, default 4)\n"
       << "\n"
       << "Camera capture (independent of model/image input):\n"
       << "  --camera <path>       V4L2 device (e.g. /dev/video0)\n"
       << "  --capture-only        Capture frames without model inference\n"
       << "  --max-frames <n>      Frames to capture (default 30)\n"
       << "  --camera-format <f>   MJPG or YUYV (default YUYV)\n"
       << "  --camera-width <n>    Requested width (default 640)\n"
       << "  --camera-height <n>   Requested height (default 480)\n"
       << "  --camera-fps <n>      Requested fps (default 30)\n"
       << "  --camera-timeout-ms <n> Read timeout in ms (default 2000)\n"
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

bool parseSizeOption(
    const std::string& name,
    const std::string& value,
    std::size_t& output,
    std::string& error)
{
    std::size_t first = 0;
    while (first < value.size() &&
           std::isspace(static_cast<unsigned char>(value[first])) != 0) {
        ++first;
    }
    // stoull("-1") wraps to a huge unsigned value; reject the sign before
    // conversion so a typo cannot silently disable backpressure.
    if (first == value.size() || value[first] == '-') {
        error = name + " must be greater than 0";
        return false;
    }
    try {
        std::size_t consumed = 0;
        const unsigned long long parsed = std::stoull(value, &consumed);
        if (consumed != value.size()) {
            error = "invalid value for " + name + ": '" + value + "'";
            return false;
        }
        if (parsed == 0) {
            error = name + " must be greater than 0";
            return false;
        }
        if (parsed > std::numeric_limits<std::size_t>::max()) {
            error = name + " is too large";
            return false;
        }
        output = static_cast<std::size_t>(parsed);
        return true;
    } catch (const std::exception&) {
        error = "invalid value for " + name + ": '" + value + "'";
        return false;
    }
}

bool parsePositiveIntOption(
    const std::string& name,
    const std::string& value,
    int& output,
    std::string& error)
{
    std::size_t parsed = 0;
    if (!parseSizeOption(name, value, parsed, error)) {
        return false;
    }
    if (parsed > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        error = name + " is too large";
        return false;
    }
    output = static_cast<int>(parsed);
    return true;
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
    //
    // Resolve the parent directory FIRST. A relative target such as
    // "../result.json" must be interpreted relative to the real parent, not a
    // lexical path whose parent may itself be a symlink.
    for (int depth = 0; depth < 40; ++depth) {
        const std::filesystem::path parent = absolute.parent_path();
        const std::filesystem::path filename = absolute.filename();
        error_code.clear();
        std::filesystem::path resolved_parent = std::filesystem::weakly_canonical(
            parent.empty() ? std::filesystem::path(".") : parent, error_code);
        if (error_code) {
            error_code.clear();
            resolved_parent = parent;
        }
        absolute = resolved_parent / filename;

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
            absolute = absolute.parent_path() / target;
        } else {
            absolute = target;
        }
        // Do not lexically_normal() here: the next iteration must resolve the
        // parent (including "..") in the real filesystem namespace.
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
        } else if (option == "--queue-capacity") {
            std::string value;
            if (!take_value("--queue-capacity", value) ||
                !parseSizeOption("--queue-capacity", value,
                                 options.queue_capacity, error)) {
                return false;
            }
            options.queue_capacity_set = true;
        } else if (option == "--camera") {
            if (!take_value("--camera", options.camera_path)) {
                return false;
            }
        } else if (option == "--capture-only") {
            if (has_inline_value) {
                error = "--capture-only does not take a value";
                return false;
            }
            options.capture_only = true;
        } else if (option == "--max-frames") {
            std::string value;
            if (!take_value("--max-frames", value) ||
                !parseSizeOption("--max-frames", value,
                                 options.max_frames, error)) {
                return false;
            }
            options.max_frames_set = true;
            options.camera_options_set = true;
        } else if (option == "--camera-format") {
            std::string value;
            if (!take_value("--camera-format", value)) {
                return false;
            }
            const std::string normalized =
                edge_ai::normalizeCameraFormat(value);
            if (normalized.empty()) {
                error = "unsupported camera format: '" + value +
                        "' (supported: MJPG, YUYV)";
                return false;
            }
            options.camera_format = normalized;
            options.camera_options_set = true;
        } else if (option == "--camera-width") {
            std::string value;
            if (!take_value("--camera-width", value) ||
                !parsePositiveIntOption("--camera-width", value,
                                        options.camera_width, error)) {
                return false;
            }
            options.camera_options_set = true;
        } else if (option == "--camera-height") {
            std::string value;
            if (!take_value("--camera-height", value) ||
                !parsePositiveIntOption("--camera-height", value,
                                        options.camera_height, error)) {
                return false;
            }
            options.camera_options_set = true;
        } else if (option == "--camera-fps") {
            std::string value;
            if (!take_value("--camera-fps", value) ||
                !parsePositiveIntOption("--camera-fps", value,
                                        options.camera_fps, error)) {
                return false;
            }
            options.camera_options_set = true;
        } else if (option == "--camera-timeout-ms") {
            std::string value;
            if (!take_value("--camera-timeout-ms", value) ||
                !parsePositiveIntOption("--camera-timeout-ms", value,
                                        options.camera_timeout_ms, error)) {
                return false;
            }
            options.camera_options_set = true;
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

    const bool has_camera = !options.camera_path.empty();
    if (options.capture_only) {
        if (!has_camera) {
            error = "--capture-only requires --camera";
            return false;
        }
        if (!options.model_path.empty() || !options.image_path.empty() ||
            !options.image_dir.empty()) {
            error = "--capture-only cannot be combined with model/image inputs";
            return false;
        }
        if (!options.json_out.empty() || !options.dump_input.empty()) {
            error = "--capture-only does not support --json-out/--dump-input";
            return false;
        }
        if (options.queue_capacity_set) {
            error = "--queue-capacity is for batch inference, not capture-only";
            return false;
        }
        return true;
    }
    if (has_camera) {
        error = "--camera requires --capture-only";
        return false;
    }
    if (options.camera_options_set) {
        error = "camera options require --capture-only";
        return false;
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
    if (options.queue_capacity_set && !has_image_dir) {
        error = "--queue-capacity is only supported with --image-dir";
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

long long timestampMillis(edge_ai::FrameTimestamp timestamp)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               timestamp.time_since_epoch())
        .count();
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
    edge_ai::FrameTimestamp timestamp,
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
    os << "    \"timestamp_unix_ms\": ";
    if (timestamp == edge_ai::FrameTimestamp{}) {
        os << "null";
    } else {
        os << timestampMillis(timestamp);
    }
    os << ",\n";
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
        os << "      \"index\": " << frame.index << ",\n";
        os << "      \"status\": \"" << status << "\",\n";
        os << "      \"error\": \"" << jsonEscape(frame.error) << "\",\n";
        os << "      \"width\": " << frame.width << ",\n";
        os << "      \"height\": " << frame.height << ",\n";
        os << "      \"timestamp_unix_ms\": ";
        if (frame.timestamp == edge_ai::FrameTimestamp{}) {
            os << "null";
        } else {
            os << timestampMillis(frame.timestamp);
        }
        os << ",\n";
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
        if (!frame.quality_available) {
            // A failed frame has no quality decision. Emitting
            // is_ng:false here could be mistaken for a completed OK result.
            os << "      \"quality\": null\n";
        } else {
            os << "      \"quality\": {\"status\": \"" << status
               << "\", \"is_ng\": "
               << (frame.quality.is_ng ? "true" : "false")
               << ", \"confidence_threshold\": " << options.conf_threshold
               << ", \"total_detections\": " << frame.quality.total_detections
               << ", \"qualifying_defects\": " << frame.quality.defects.size()
               << "}\n";
        }
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
                         frame.timestamp, error)) {
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
    edge_ai::PipelineOptions pipeline_options;
    pipeline_options.input_queue_capacity = options.queue_capacity;
    pipeline_options.result_queue_capacity = options.queue_capacity;
    edge_ai::ThreadedPipeline pipeline(source, *runner, postprocess_options,
                                      pipeline_options);

    printModelInfo(runner->info());
    std::cout << "batch input: " << options.image_dir << "\n";
    std::cout << "quality rule: NG if at least one detection has confidence > "
              << options.conf_threshold << "\n";

    const edge_ai::ResultCallback callback =
        [](const edge_ai::FrameResult& frame, std::string&) {
            if (!frame.error.empty()) {
                std::cout << "[ERROR] " << frame.source << ": "
                          << frame.error << "\n";
            } else {
                std::cout << "["
                          << edge_ai::qualityLabel(frame.quality.is_ng) << "] "
                          << frame.source
                          << " detections=" << frame.detections.size()
                          << "\n";
            }
            for (const auto& anomaly : frame.anomalies) {
                std::cerr << "warning: " << frame.source << ": " << anomaly
                          << "\n";
            }
            return true;
        };

    edge_ai::BatchResult batch;
    edge_ai::PipelineStats pipeline_stats;
    std::string fatal_error;
    if (!pipeline.run(batch, pipeline_stats, callback, fatal_error)) {
        std::cerr << kProgram << ": " << fatal_error << "\n";
        return 1;
    }
    batch.skipped_unsupported = source.skippedUnsupported();

    std::cout << "threads: reader=" << pipeline_stats.reader_thread
              << " inference=" << pipeline_stats.inference_thread
              << " result=" << pipeline_stats.result_thread << "\n";
    std::cout << "queues: input capacity="
              << pipeline_stats.input_queue_capacity
              << " max_depth=" << pipeline_stats.input_queue_stats.max_depth
              << " producer_waits="
              << pipeline_stats.input_queue_stats.producer_wait_count
              << " consumer_waits="
              << pipeline_stats.input_queue_stats.consumer_wait_count
              << "; result capacity="
              << pipeline_stats.result_queue_capacity
              << " max_depth=" << pipeline_stats.result_queue_stats.max_depth
              << " producer_waits="
              << pipeline_stats.result_queue_stats.producer_wait_count
              << " consumer_waits="
              << pipeline_stats.result_queue_stats.consumer_wait_count << "\n";
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

int runCapture(const Options& options)
{
    std::string error;
    edge_ai::CameraOptions camera_options;
    camera_options.device = options.camera_path;
    camera_options.format = options.camera_format;
    camera_options.width = options.camera_width;
    camera_options.height = options.camera_height;
    camera_options.fps = options.camera_fps;
    camera_options.timeout_ms = options.camera_timeout_ms;

    edge_ai::CameraSource camera(camera_options, error);
    if (!error.empty()) {
        std::cerr << kProgram << ": " << error << "\n";
        return 1;
    }

    const edge_ai::CameraInfo& info = camera.info();
    std::cout << "camera: " << info.device << "\n";
    std::cout << "requested: " << info.requested_format << " "
              << info.requested_width << "x" << info.requested_height
              << " @ " << info.requested_fps << " fps timeout="
              << options.camera_timeout_ms << " ms\n";
    std::cout << "actual: " << info.actual_format << " "
              << info.actual_width << "x" << info.actual_height << " @ "
              << std::fixed << std::setprecision(3) << info.actual_fps
              << std::defaultfloat << " fps bytes_per_line="
              << info.bytes_per_line << "\n";

    std::size_t captured = 0;
    std::size_t first_index = 0;
    std::size_t last_index = 0;
    for (std::size_t i = 0; i < options.max_frames; ++i) {
        edge_ai::Frame frame;
        const edge_ai::SourceStatus status = camera.next(frame, error);
        if (status != edge_ai::SourceStatus::Frame) {
            std::cerr << kProgram << ": camera capture failed: "
                      << (error.empty() ? "camera ended early" : error)
                      << "\n";
            return 1;
        }
        if (captured == 0) {
            first_index = frame.index;
        }
        last_index = frame.index;
        ++captured;
        const auto timestamp_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                frame.timestamp.time_since_epoch())
                .count();
        std::cout << "frame " << frame.index << ": " << frame.image.cols
                  << "x" << frame.image.rows << " channels="
                  << frame.image.channels() << " timestamp_ms="
                  << timestamp_ms << "\n";
    }

    std::cout << "camera capture complete: device=" << info.device
              << " format=" << info.actual_format
              << " size=" << info.actual_width << "x" << info.actual_height
              << " fps=" << info.actual_fps
              << " frames=" << captured << " index_range="
              << first_index << ".." << last_index << "\n";
    return 0;
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
    if (options.capture_only) {
        return runCapture(options);
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

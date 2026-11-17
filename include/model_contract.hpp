#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace edge_ai {

// Contract verified in task 2A for neu_det_yolov8n_best.onnx.
constexpr int kInputSize = 640;
constexpr int kNumPredictions = 8400;
constexpr int kNumClasses = 6;
constexpr int kOutputChannels = 4 + kNumClasses;  // 10

struct TensorContract {
    std::string name;
    std::vector<int64_t> shape;
    std::string type;  // e.g. "tensor(float)"
};

struct ModelContract {
    std::size_t input_count = 0;
    std::size_t output_count = 0;
    TensorContract input;
    TensorContract output;
    std::map<int, std::string> class_names;
};

// Returns an empty string when the contract matches the verified model.
// Otherwise returns a diagnostic naming the expected and actual value.
std::string validateModelContract(const ModelContract& contract);

std::string formatShape(const std::vector<int64_t>& shape);

// Parses the ONNX metadata string "{0: 'crazing', 1: 'inclusion', ...}".
// Rejects missing/duplicate/non-contiguous ids instead of guessing a mapping.
bool parseClassNamesMetadata(
    const std::string& raw,
    std::map<int, std::string>& class_names,
    std::string& error);

}  // namespace edge_ai

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "image_preprocess.hpp"
#include "postprocess.hpp"

namespace edge_ai {

struct ModelInfo {
    std::string path;
    std::string sha256;
    std::string ort_version;
    std::vector<std::string> providers;
    std::vector<std::string> available_providers;
    std::size_t input_count = 0;
    std::size_t output_count = 0;
    std::string input_name;
    std::vector<int64_t> input_shape;
    std::string input_type;
    std::string output_name;
    std::vector<int64_t> output_shape;
    std::string output_type;
    std::map<int, std::string> class_names;
};

class ModelRunner {
public:
    ~ModelRunner();
    ModelRunner(const ModelRunner&) = delete;
    ModelRunner& operator=(const ModelRunner&) = delete;

    // Returns nullptr and sets `error` on missing/corrupt/incompatible models.
    static std::unique_ptr<ModelRunner> load(
        const std::string& model_path,
        std::string& error);

    const ModelInfo& info() const;

    PreprocessedInput preprocess(const cv::Mat& image) const;

    // Validates the runtime output shape/type before postprocessing.
    PostprocessResult infer(
        const PreprocessedInput& input,
        const PostprocessOptions& options) const;

private:
    struct Impl;
    explicit ModelRunner(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace edge_ai

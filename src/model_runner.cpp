#include "model_runner.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "model_contract.hpp"
#include "sha256.hpp"

#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>

namespace edge_ai {
namespace {

std::string elementTypeToString(ONNXTensorElementDataType type)
{
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return "tensor(float)";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return "tensor(double)";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return "tensor(float16)";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return "tensor(int8)";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return "tensor(uint8)";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return "tensor(int32)";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return "tensor(int64)";
        default:
            return "tensor(type=" + std::to_string(static_cast<int>(type)) + ")";
    }
}

std::vector<int64_t> tensorShape(
    const Ort::Session& session,
    bool input,
    std::size_t index)
{
    Ort::TypeInfo type_info = input ? session.GetInputTypeInfo(index)
                                    : session.GetOutputTypeInfo(index);
    return type_info.GetTensorTypeAndShapeInfo().GetShape();
}

std::string tensorType(
    const Ort::Session& session,
    bool input,
    std::size_t index)
{
    Ort::TypeInfo type_info = input ? session.GetInputTypeInfo(index)
                                    : session.GetOutputTypeInfo(index);
    return elementTypeToString(
        type_info.GetTensorTypeAndShapeInfo().GetElementType());
}

std::string tensorName(
    const Ort::Session& session,
    bool input,
    std::size_t index,
    Ort::AllocatorWithDefaultOptions& allocator)
{
    auto name = input ? session.GetInputNameAllocated(index, allocator)
                      : session.GetOutputNameAllocated(index, allocator);
    return name ? std::string(name.get()) : std::string{};
}

}  // namespace

struct ModelRunner::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "edge_ai"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    ModelInfo info;
};

ModelRunner::ModelRunner(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

ModelRunner::~ModelRunner() = default;

const ModelInfo& ModelRunner::info() const
{
    return impl_->info;
}

std::unique_ptr<ModelRunner> ModelRunner::load(
    const std::string& model_path,
    std::string& error)
{
    error.clear();
    std::error_code fs_error;
    if (!std::filesystem::exists(model_path, fs_error) || fs_error) {
        error = "model file not found: " + model_path;
        return nullptr;
    }
    if (!std::filesystem::is_regular_file(model_path, fs_error) || fs_error) {
        error = "model path is not a regular file: " + model_path;
        return nullptr;
    }

    std::string sha_error;
    const std::string model_sha = sha256File(model_path, sha_error);
    if (model_sha.empty()) {
        error = "failed to compute model SHA256: " + sha_error;
        return nullptr;
    }

    std::unique_ptr<Impl> impl;
    try {
        impl = std::make_unique<Impl>();
        impl->options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        impl->options.AppendExecutionProvider_CPU(0);
        impl->session = std::make_unique<Ort::Session>(
            impl->env, model_path.c_str(), impl->options);
    } catch (const Ort::Exception& exception) {
        error = "failed to load ONNX model '" + model_path + "': " +
                exception.what();
        return nullptr;
    } catch (const std::exception& exception) {
        error = "failed to initialize ONNX Runtime: " +
                std::string(exception.what());
        return nullptr;
    }

    try {
        Ort::AllocatorWithDefaultOptions allocator;
        ModelContract contract;
        contract.input_count = impl->session->GetInputCount();
        contract.output_count = impl->session->GetOutputCount();
        if (contract.input_count > 0) {
            contract.input.name = tensorName(*impl->session, true, 0, allocator);
            contract.input.shape = tensorShape(*impl->session, true, 0);
            contract.input.type = tensorType(*impl->session, true, 0);
        }
        if (contract.output_count > 0) {
            contract.output.name = tensorName(*impl->session, false, 0, allocator);
            contract.output.shape = tensorShape(*impl->session, false, 0);
            contract.output.type = tensorType(*impl->session, false, 0);
        }

        Ort::ModelMetadata metadata = impl->session->GetModelMetadata();
        auto raw_names =
            metadata.LookupCustomMetadataMapAllocated("names", allocator);
        if (!raw_names) {
            error = "model contract mismatch: metadata has no 'names' entry";
            return nullptr;
        }
        std::string parse_error;
        if (!parseClassNamesMetadata(raw_names.get(), contract.class_names,
                                     parse_error)) {
            error = "model contract mismatch: invalid class metadata: " +
                    parse_error;
            return nullptr;
        }

        const std::string contract_error = validateModelContract(contract);
        if (!contract_error.empty()) {
            error = "model contract mismatch: " + contract_error;
            return nullptr;
        }

        ModelInfo& info = impl->info;
        info.path = model_path;
        info.sha256 = model_sha;
        info.ort_version = Ort::GetVersionString();
        info.providers = {"CPUExecutionProvider"};
        info.available_providers = Ort::GetAvailableProviders();
        if (std::find(info.available_providers.begin(),
                      info.available_providers.end(),
                      "CPUExecutionProvider") ==
            info.available_providers.end()) {
            error = "ONNX Runtime CPUExecutionProvider is unavailable";
            return nullptr;
        }
        info.input_count = contract.input_count;
        info.output_count = contract.output_count;
        info.input_name = contract.input.name;
        info.input_shape = contract.input.shape;
        info.input_type = contract.input.type;
        info.output_name = contract.output.name;
        info.output_shape = contract.output.shape;
        info.output_type = contract.output.type;
        info.class_names = std::move(contract.class_names);
    } catch (const Ort::Exception& exception) {
        error = "failed to inspect ONNX model '" + model_path + "': " +
                exception.what();
        return nullptr;
    } catch (const std::exception& exception) {
        error = "failed to inspect ONNX model '" + model_path + "': " +
                std::string(exception.what());
        return nullptr;
    }

    return std::unique_ptr<ModelRunner>(
        new ModelRunner(std::move(impl)));
}

PreprocessedInput ModelRunner::preprocess(const cv::Mat& image) const
{
    return preprocessImage(image, kInputSize);
}

PostprocessResult ModelRunner::infer(
    const PreprocessedInput& input,
    const PostprocessOptions& options) const
{
    PostprocessResult result;
    const std::size_t expected_input_size =
        static_cast<std::size_t>(3) * kInputSize * kInputSize;
    if (input.data.size() != expected_input_size) {
        result.error = "preprocessed tensor size mismatch: expected " +
                       std::to_string(expected_input_size) + ", got " +
                       std::to_string(input.data.size());
        return result;
    }

    try {
        std::array<int64_t, 4> shape = {1, 3, kInputSize, kInputSize};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        // ONNX Runtime's C++ API takes a non-const pointer but does not modify
        // the input buffer during Run().
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(input.data.data()),
            input.data.size(), shape.data(), shape.size());

        const char* input_names[] = {impl_->info.input_name.c_str()};
        const char* output_names[] = {impl_->info.output_name.c_str()};
        std::vector<Ort::Value> outputs = impl_->session->Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
            output_names, 1);

        if (outputs.size() != 1) {
            result.error = "runtime output count mismatch: expected 1, got " +
                           std::to_string(outputs.size());
            return result;
        }

        Ort::TensorTypeAndShapeInfo output_info =
            outputs[0].GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> output_shape = output_info.GetShape();
        const std::vector<int64_t> expected_shape = {
            1, kOutputChannels, kNumPredictions};
        if (output_shape != expected_shape) {
            result.error = "runtime output shape mismatch: expected " +
                           formatShape(expected_shape) + ", got " +
                           formatShape(output_shape);
            return result;
        }
        if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            result.error =
                "runtime output type mismatch: expected tensor(float), got " +
                elementTypeToString(output_info.GetElementType());
            return result;
        }

        const std::size_t count = output_info.GetElementCount();
        if (count != static_cast<std::size_t>(kOutputChannels) * kNumPredictions) {
            result.error = "runtime output element count mismatch: expected " +
                           std::to_string(static_cast<std::size_t>(kOutputChannels) *
                                          kNumPredictions) +
                           ", got " + std::to_string(count);
            return result;
        }

        const float* raw = outputs[0].GetTensorData<float>();
        std::vector<float> output(raw, raw + count);
        return postprocess(output, impl_->info.class_names, input.letterbox,
                           options);
    } catch (const Ort::Exception& exception) {
        result.error = std::string("inference failed: ") + exception.what();
    } catch (const std::exception& exception) {
        result.error = std::string("inference failed: ") + exception.what();
    }
    return result;
}

}  // namespace edge_ai

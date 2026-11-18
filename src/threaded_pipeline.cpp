#include "threaded_pipeline.hpp"

#include <atomic>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "quality.hpp"

namespace edge_ai {
namespace {

struct InputItem {
    Frame frame;
    std::string error;
    bool has_error = false;
};

}  // namespace

ThreadedPipeline::ThreadedPipeline(
    ImageSource& source,
    const ModelRunner& runner,
    const PostprocessOptions& options,
    const PipelineOptions& pipeline_options)
    : source_(source),
      runner_(runner),
      options_(options),
      pipeline_options_(pipeline_options)
{
    if (pipeline_options_.input_queue_capacity == 0 ||
        pipeline_options_.result_queue_capacity == 0) {
        throw std::invalid_argument(
            "queue capacities must be greater than 0");
    }
}

bool ThreadedPipeline::run(
    BatchResult& result,
    PipelineStats& stats,
    const ResultCallback& callback,
    std::string& fatal_error)
{
    fatal_error.clear();
    result = BatchResult{};
    stats = PipelineStats{};
    stats.input_queue_capacity = pipeline_options_.input_queue_capacity;
    stats.result_queue_capacity = pipeline_options_.result_queue_capacity;

    BoundedQueue<InputItem> input_queue(
        pipeline_options_.input_queue_capacity);
    BoundedQueue<FrameResult> result_queue(
        pipeline_options_.result_queue_capacity);
    if (pipeline_options_.input_producer_wait_hook) {
        input_queue.setProducerWaitCallback(
            pipeline_options_.input_producer_wait_hook);
    }
    if (pipeline_options_.input_producer_wake_hook) {
        input_queue.setProducerWakeCallback(
            pipeline_options_.input_producer_wake_hook);
    }
    if (pipeline_options_.result_producer_wait_hook) {
        result_queue.setProducerWaitCallback(
            pipeline_options_.result_producer_wait_hook);
    }
    if (pipeline_options_.result_producer_wake_hook) {
        result_queue.setProducerWakeCallback(
            pipeline_options_.result_producer_wake_hook);
    }

    std::atomic<bool> stop{false};
    std::mutex fatal_mutex;
    std::string fatal_reason;
    const auto request_stop = [&](const std::string& reason) {
        {
            std::lock_guard<std::mutex> lock(fatal_mutex);
            if (fatal_reason.empty()) {
                fatal_reason = reason;
            }
        }
        stop.store(true);
        input_queue.close();
        result_queue.close();
    };

    BatchResult batch;

    std::function<void()> reader_task = [&] {
        stats.reader_thread = std::this_thread::get_id();
        try {
            while (!stop.load()) {
                Frame frame;
                std::string error;
                const SourceStatus status = source_.next(frame, error);
                if (status == SourceStatus::End) {
                    break;
                }
                InputItem item;
                item.frame.source = frame.source.empty()
                                        ? "<unknown source>"
                                        : frame.source;
                item.frame.index = frame.index;
                item.frame.timestamp = frame.timestamp;
                if (status == SourceStatus::Frame) {
                    item.frame.image = std::move(frame.image);
                } else {
                    item.has_error = true;
                    item.error = error;
                }
                if (!input_queue.push(std::move(item))) {
                    break;
                }
            }
        } catch (const std::exception& exception) {
            request_stop(
                std::string("image source exception: ") + exception.what());
        } catch (...) {
            request_stop("image source exception: unknown");
        }
        input_queue.close();
    };

    std::function<void()> inference_task = [&] {
        stats.inference_thread = std::this_thread::get_id();
        while (!stop.load()) {
            InputItem item;
            if (input_queue.pop(item) == PopStatus::Closed) {
                break;
            }
            if (stop.load()) {
                break;
            }

            FrameResult frame_result;
            frame_result.source = item.frame.source;
            frame_result.index = item.frame.index;
            frame_result.timestamp = item.frame.timestamp;
            if (item.has_error) {
                frame_result.error = item.error;
            } else {
                frame_result.width = item.frame.image.cols;
                frame_result.height = item.frame.image.rows;
                try {
                    const PreprocessedInput input =
                        runner_.preprocess(item.frame.image);
                    frame_result.letterbox = input.letterbox;
                    const PostprocessResult postprocessed =
                        runner_.infer(input, options_);
                    if (!postprocessed.error.empty()) {
                        frame_result.error = postprocessed.error;
                    } else {
                        QualityResult quality;
                        std::string quality_error;
                        if (!evaluateQuality(postprocessed.detections,
                                             options_.conf_threshold, quality,
                                             quality_error)) {
                            frame_result.error = quality_error;
                        } else {
                            frame_result.inference_ok = true;
                            frame_result.detections =
                                postprocessed.detections;
                            frame_result.anomalies = postprocessed.anomalies;
                            frame_result.quality_available = true;
                            frame_result.quality = quality;
                        }
                    }
                } catch (const std::exception& exception) {
                    frame_result.error =
                        std::string("frame inference failed: ") +
                        exception.what();
                }
            }

            if (!result_queue.push(std::move(frame_result))) {
                break;
            }
        }
        result_queue.close();
    };

    std::function<void()> result_task = [&] {
        stats.result_thread = std::this_thread::get_id();
        while (!stop.load()) {
            FrameResult frame_result;
            if (result_queue.pop(frame_result) == PopStatus::Closed) {
                break;
            }
            if (stop.load()) {
                break;
            }

            if (frame_result.error.empty()) {
                if (frame_result.quality.is_ng) {
                    ++batch.ng_count;
                } else {
                    ++batch.ok_count;
                }
            } else {
                ++batch.error_count;
            }
            batch.frames.push_back(frame_result);

            if (callback) {
                std::string callback_error;
                bool accepted = false;
                try {
                    accepted = callback(frame_result, callback_error);
                } catch (const std::exception& exception) {
                    callback_error = exception.what();
                } catch (...) {
                    callback_error = "unknown exception";
                }
                if (!accepted) {
                    request_stop(
                        "result callback failed: " +
                        (callback_error.empty() ? "unspecified error"
                                                : callback_error));
                    break;
                }
            }
        }
    };

    const auto finalize = [&](const std::string& forced_error) {
        stats.input_queue_stats = input_queue.stats();
        stats.result_queue_stats = result_queue.stats();
        stats.total = batch.frames.size();
        stats.ok = batch.ok_count;
        stats.ng = batch.ng_count;
        stats.errors = batch.error_count;
        result = std::move(batch);
        if (!forced_error.empty()) {
            fatal_error = forced_error;
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(fatal_mutex);
            if (!fatal_reason.empty()) {
                fatal_error = fatal_reason;
                return false;
            }
        }
        return true;
    };

    std::thread reader_thread;
    std::thread inference_thread;
    std::thread result_thread;
    std::string start_error;

    const auto start_stage = [&](PipelineStage stage,
                                 std::function<void()> task,
                                 std::thread& target) {
        try {
            if (pipeline_options_.thread_factory) {
                target = pipeline_options_.thread_factory(
                    stage, std::move(task));
            } else {
                target = std::thread(std::move(task));
            }
            if (!target.joinable()) {
                start_error = "thread factory returned a non-joinable thread";
                return false;
            }
            return true;
        } catch (const std::exception& exception) {
            start_error = exception.what();
        } catch (...) {
            start_error = "unknown exception";
        }
        return false;
    };

    if (!start_stage(PipelineStage::Reader, std::move(reader_task),
                     reader_thread)) {
        const std::string message =
            "failed to start reader thread: " + start_error;
        request_stop(message);
        return finalize(message);
    }
    if (!start_stage(PipelineStage::Inference, std::move(inference_task),
                     inference_thread)) {
        const std::string message =
            "failed to start inference thread: " + start_error;
        request_stop(message);
        if (reader_thread.joinable()) {
            reader_thread.join();
        }
        return finalize(message);
    }
    if (!start_stage(PipelineStage::Result, std::move(result_task),
                     result_thread)) {
        const std::string message =
            "failed to start result thread: " + start_error;
        request_stop(message);
        if (reader_thread.joinable()) {
            reader_thread.join();
        }
        if (inference_thread.joinable()) {
            inference_thread.join();
        }
        return finalize(message);
    }

    reader_thread.join();
    inference_thread.join();
    result_thread.join();
    return finalize("");
}

}  // namespace edge_ai

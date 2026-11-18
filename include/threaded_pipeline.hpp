#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <thread>

#include "batch_processor.hpp"
#include "bounded_queue.hpp"
#include "image_source.hpp"
#include "model_runner.hpp"

namespace edge_ai {

enum class PipelineStage { Reader, Inference, Result };

// Test/instrumentation hook allowing a controlled thread-start failure.
// The production path leaves this empty and uses std::thread.
using ThreadFactory =
    std::function<std::thread(PipelineStage, std::function<void()>)>;

struct PipelineOptions {
    std::size_t input_queue_capacity = 4;
    std::size_t result_queue_capacity = 4;
    // Invoked just before the corresponding producer blocks on a full queue.
    std::function<void()> input_producer_wait_hook;
    std::function<void()> result_producer_wait_hook;
    // Optional injection point for thread-start failure tests.
    ThreadFactory thread_factory;
};

struct PipelineStats {
    std::size_t input_queue_capacity = 0;
    std::size_t result_queue_capacity = 0;
    QueueStats input_queue_stats;
    QueueStats result_queue_stats;
    std::thread::id reader_thread;
    std::thread::id inference_thread;
    std::thread::id result_thread;
    std::size_t total = 0;
    std::size_t ok = 0;
    std::size_t ng = 0;
    std::size_t errors = 0;
};

// Called in result order by the result thread. Returning false (or throwing)
// is a fatal downstream error: queues are closed, all threads join, and run()
// returns false with a diagnostic.
using ResultCallback = std::function<bool(const FrameResult&, std::string&)>;

class ThreadedPipeline {
public:
    ThreadedPipeline(
        ImageSource& source,
        const ModelRunner& runner,
        const PostprocessOptions& options,
        const PipelineOptions& pipeline_options = {});

    ThreadedPipeline(const ThreadedPipeline&) = delete;
    ThreadedPipeline& operator=(const ThreadedPipeline&) = delete;

    // Reader -> input queue -> inference -> result queue -> result thread.
    // A source exception is fatal because it may not have advanced; per-frame
    // image/inference failures remain ordinary ERROR results.
    bool run(
        BatchResult& result,
        PipelineStats& stats,
        const ResultCallback& callback,
        std::string& fatal_error);

private:
    ImageSource& source_;
    const ModelRunner& runner_;
    PostprocessOptions options_;
    PipelineOptions pipeline_options_;
};

}  // namespace edge_ai

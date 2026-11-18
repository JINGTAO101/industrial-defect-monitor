#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "batch_processor.hpp"
#include "model_runner.hpp"
#include "threaded_pipeline.hpp"

using namespace std::chrono_literals;

#ifndef EDGE_AI_FIXTURE_DIR
#define EDGE_AI_FIXTURE_DIR "tests/fixtures"
#endif

namespace {

using TestFunction = bool (*)(std::string&);

struct TestCase {
    const char* name;
    TestFunction function;
};

bool expect(bool condition, const std::string& message, std::string& error)
{
    if (!condition) {
        error = message;
        return false;
    }
    return true;
}

class SequenceSource final : public edge_ai::ImageSource {
public:
    explicit SequenceSource(int count) : count_(count) {}

    edge_ai::SourceStatus next(edge_ai::Frame& frame,
                               std::string&) override
    {
        if (next_ >= count_) {
            return edge_ai::SourceStatus::End;
        }
        frame.image = cv::Mat(200, 320, CV_8UC3, cv::Scalar(0, 0, 0));
        frame.source = "frame-" + std::to_string(next_);
        frame.index = static_cast<std::size_t>(next_);
        frame.timestamp = std::chrono::system_clock::now();
        ++next_;
        return edge_ai::SourceStatus::Frame;
    }

private:
    int count_;
    int next_ = 0;
};

struct ProducerWaitSignals {
    std::atomic<bool> input_seen{false};
    std::atomic<bool> result_seen{false};
    std::promise<void> input_waiting;
    std::promise<void> result_waiting;
    std::future<void> input_future = input_waiting.get_future();
    std::future<void> result_future = result_waiting.get_future();

    std::function<void()> input_callback()
    {
        return [this] {
            bool expected = false;
            if (input_seen.compare_exchange_strong(expected, true)) {
                input_waiting.set_value();
            }
        };
    }

    std::function<void()> result_callback()
    {
        return [this] {
            bool expected = false;
            if (result_seen.compare_exchange_strong(expected, true)) {
                result_waiting.set_value();
            }
        };
    }
};

bool distinctThreads(const edge_ai::PipelineStats& stats)
{
    return stats.reader_thread != std::thread::id{} &&
           stats.inference_thread != std::thread::id{} &&
           stats.result_thread != std::thread::id{} &&
           stats.reader_thread != stats.inference_thread &&
           stats.reader_thread != stats.result_thread &&
           stats.inference_thread != stats.result_thread;
}

bool loadRunner(
    std::unique_ptr<edge_ai::ModelRunner>& runner,
    std::string& error)
{
    runner = edge_ai::ModelRunner::load(
        std::string(EDGE_AI_FIXTURE_DIR) + "/zero_output.onnx", error);
    return runner != nullptr;
}

bool testNormalDrain(std::string& error)
{
    std::unique_ptr<edge_ai::ModelRunner> runner;
    std::string load_error;
    if (!expect(loadRunner(runner, load_error),
                "cannot load zero_output.onnx: " + load_error, error)) {
        return false;
    }

    SequenceSource source(4);
    edge_ai::PipelineOptions pipeline_options;
    pipeline_options.input_queue_capacity = 2;
    pipeline_options.result_queue_capacity = 2;
    edge_ai::ThreadedPipeline pipeline(
        source, *runner, edge_ai::PostprocessOptions{}, pipeline_options);
    std::vector<std::size_t> callback_indices;
    const edge_ai::ResultCallback callback =
        [&](const edge_ai::FrameResult& frame, std::string&) {
            callback_indices.push_back(frame.index);
            return true;
        };

    edge_ai::BatchResult batch;
    edge_ai::PipelineStats stats;
    std::string fatal_error;
    if (!expect(pipeline.run(batch, stats, callback, fatal_error),
                "normal pipeline failed: " + fatal_error, error) ||
        !expect(batch.frames.size() == 4 && batch.ok_count == 4 &&
                    batch.ng_count == 0 && batch.error_count == 0,
                "normal pipeline counts are wrong", error) ||
        !expect(callback_indices.size() == 4,
                "normal pipeline callback count is wrong", error)) {
        return false;
    }
    for (std::size_t i = 0; i < callback_indices.size(); ++i) {
        if (!expect(callback_indices[i] == i,
                    "normal pipeline result order is wrong", error)) {
            return false;
        }
    }
    if (!expect(distinctThreads(stats),
                "pipeline did not use three distinct threads", error) ||
        !expect(stats.input_queue_stats.max_depth <= 2 &&
                    stats.result_queue_stats.max_depth <= 2,
                "normal pipeline exceeded queue capacity", error)) {
        return false;
    }

    std::cout << "normal pipeline: threads(reader=" << stats.reader_thread
              << ", inference=" << stats.inference_thread
              << ", result=" << stats.result_thread << ") "
              << "queues(input cap=" << stats.input_queue_capacity
              << " max=" << stats.input_queue_stats.max_depth
              << ", result cap=" << stats.result_queue_capacity
              << " max=" << stats.result_queue_stats.max_depth << ") "
              << "counts(total=" << stats.total << " ok=" << stats.ok
              << " ng=" << stats.ng << " errors=" << stats.errors << ")\n";
    return true;
}

bool testFullBackpressure(std::string& error)
{
    std::unique_ptr<edge_ai::ModelRunner> runner;
    std::string load_error;
    if (!expect(loadRunner(runner, load_error),
                "cannot load zero_output.onnx: " + load_error, error)) {
        return false;
    }

    SequenceSource source(5);
    ProducerWaitSignals wait_signals;
    edge_ai::PipelineOptions pipeline_options;
    pipeline_options.input_queue_capacity = 1;
    pipeline_options.result_queue_capacity = 1;
    pipeline_options.input_producer_wait_hook =
        wait_signals.input_callback();
    pipeline_options.result_producer_wait_hook =
        wait_signals.result_callback();
    edge_ai::ThreadedPipeline pipeline(
        source, *runner, edge_ai::PostprocessOptions{}, pipeline_options);

    std::promise<void> first_result;
    std::future<void> first_future = first_result.get_future();
    std::promise<void> release_result;
    std::future<void> release_future = release_result.get_future();
    std::atomic<bool> first{true};
    std::vector<std::size_t> indices;
    const edge_ai::ResultCallback callback =
        [&](const edge_ai::FrameResult& frame, std::string&) {
            bool expected = true;
            if (first.compare_exchange_strong(expected, false)) {
                first_result.set_value();
                release_future.wait();
            }
            indices.push_back(frame.index);
            return true;
        };

    edge_ai::BatchResult batch;
    edge_ai::PipelineStats stats;
    std::string fatal_error;
    bool run_ok = false;
    bool timed_out = false;
    std::thread runner_thread([&] {
        run_ok = pipeline.run(batch, stats, callback, fatal_error);
    });

    if (first_future.wait_for(2000ms) != std::future_status::ready ||
        wait_signals.input_future.wait_for(2000ms) !=
            std::future_status::ready ||
        wait_signals.result_future.wait_for(2000ms) !=
            std::future_status::ready) {
        timed_out = true;
        release_result.set_value();
    } else {
        release_result.set_value();
    }
    runner_thread.join();

    if (timed_out) {
        return expect(false,
                      "pipeline did not block both producers before release",
                      error);
    }
    if (!expect(run_ok, "backpressure pipeline failed: " + fatal_error, error) ||
        !expect(batch.frames.size() == 5 && batch.ok_count == 5 &&
                    batch.error_count == 0,
                "backpressure pipeline counts are wrong", error) ||
        !expect(indices.size() == 5,
                "backpressure pipeline result count is wrong", error)) {
        return false;
    }
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (!expect(indices[i] == i,
                    "backpressure pipeline lost, duplicated or reordered a frame",
                    error)) {
            return false;
        }
    }
    if (!expect(distinctThreads(stats),
                "backpressure pipeline did not use three threads", error) ||
        !expect(stats.input_queue_stats.max_depth == 1,
                "input queue never reached capacity 1", error) ||
        !expect(stats.result_queue_stats.max_depth == 1,
                "result queue never reached capacity 1", error) ||
        !expect(stats.input_queue_stats.producer_wait_count > 0,
                "reader never waited on a full input queue", error) ||
        !expect(stats.result_queue_stats.producer_wait_count > 0,
                "inference never waited on a full result queue", error)) {
        return false;
    }

    std::cout << "backpressure: threads(reader=" << stats.reader_thread
              << ", inference=" << stats.inference_thread
              << ", result=" << stats.result_thread << ") "
              << "queues(input cap=" << stats.input_queue_capacity
              << " max=" << stats.input_queue_stats.max_depth
              << " waits=" << stats.input_queue_stats.producer_wait_count
              << ", result cap=" << stats.result_queue_capacity
              << " max=" << stats.result_queue_stats.max_depth
              << " waits=" << stats.result_queue_stats.producer_wait_count
              << ") counts(total=" << stats.total << " ok=" << stats.ok
              << " ng=" << stats.ng << " errors=" << stats.errors << ")\n";
    return true;
}

bool testFatalWakesWaiters(std::string& error)
{
    std::unique_ptr<edge_ai::ModelRunner> runner;
    std::string load_error;
    if (!expect(loadRunner(runner, load_error),
                "cannot load zero_output.onnx: " + load_error, error)) {
        return false;
    }

    SequenceSource source(10);
    ProducerWaitSignals wait_signals;
    edge_ai::PipelineOptions pipeline_options;
    pipeline_options.input_queue_capacity = 1;
    pipeline_options.result_queue_capacity = 1;
    pipeline_options.input_producer_wait_hook =
        wait_signals.input_callback();
    pipeline_options.result_producer_wait_hook =
        wait_signals.result_callback();
    edge_ai::ThreadedPipeline pipeline(
        source, *runner, edge_ai::PostprocessOptions{}, pipeline_options);

    std::promise<void> first_result;
    std::future<void> first_future = first_result.get_future();
    std::promise<void> fail_result;
    std::future<void> fail_future = fail_result.get_future();
    std::atomic<bool> first{true};
    const edge_ai::ResultCallback callback =
        [&](const edge_ai::FrameResult&, std::string& callback_error) {
            bool expected = true;
            if (first.compare_exchange_strong(expected, false)) {
                first_result.set_value();
                fail_future.wait();
                callback_error = "injected result failure";
                return false;
            }
            return true;
        };

    edge_ai::BatchResult batch;
    edge_ai::PipelineStats stats;
    std::string fatal_error;
    bool run_ok = true;
    bool timed_out = false;
    std::thread runner_thread([&] {
        run_ok = pipeline.run(batch, stats, callback, fatal_error);
    });

    if (first_future.wait_for(2000ms) != std::future_status::ready ||
        wait_signals.input_future.wait_for(2000ms) !=
            std::future_status::ready ||
        wait_signals.result_future.wait_for(2000ms) !=
            std::future_status::ready) {
        timed_out = true;
        fail_result.set_value();
    } else {
        fail_result.set_value();
    }
    runner_thread.join();

    if (timed_out) {
        return expect(false,
                      "pipeline did not block both producers before the fatal error",
                      error);
    }
    if (!expect(!run_ok, "fatal callback did not fail the pipeline", error) ||
        !expect(fatal_error.find("result callback failed") !=
                    std::string::npos &&
                    fatal_error.find("injected result failure") !=
                        std::string::npos,
                "fatal pipeline error is not diagnostic: " + fatal_error,
                error) ||
        !expect(distinctThreads(stats),
                "fatal pipeline lost one of its three threads", error) ||
        !expect(stats.input_queue_stats.max_depth <= 1 &&
                    stats.result_queue_stats.max_depth <= 1,
                "fatal pipeline exceeded queue capacity", error) ||
        !expect(stats.input_queue_stats.producer_wait_count > 0 &&
                    stats.result_queue_stats.producer_wait_count > 0,
                "fatal pipeline did not confirm both producers blocked",
                error)) {
        return false;
    }

    std::cout << "fatal exit: threads(reader=" << stats.reader_thread
              << ", inference=" << stats.inference_thread
              << ", result=" << stats.result_thread << ") "
              << "queues(input cap=" << stats.input_queue_capacity
              << " max=" << stats.input_queue_stats.max_depth
              << ", result cap=" << stats.result_queue_capacity
              << " max=" << stats.result_queue_stats.max_depth << ") "
              << "error=\"" << fatal_error << "\"\n";
    return true;
}

bool runStartFailure(
    edge_ai::PipelineStage fail_stage,
    const std::string& stage_name,
    std::string& error)
{
    std::unique_ptr<edge_ai::ModelRunner> runner;
    std::string load_error;
    if (!expect(loadRunner(runner, load_error),
                "cannot load zero_output.onnx: " + load_error, error)) {
        return false;
    }

    SequenceSource source(5);
    edge_ai::PipelineOptions pipeline_options;
    pipeline_options.input_queue_capacity = 1;
    pipeline_options.result_queue_capacity = 1;
    pipeline_options.thread_factory =
        [fail_stage](edge_ai::PipelineStage stage,
                     std::function<void()> task) -> std::thread {
        if (stage == fail_stage) {
            throw std::system_error(
                std::make_error_code(std::errc::resource_unavailable_try_again),
                "injected thread start failure");
        }
        return std::thread(std::move(task));
    };

    edge_ai::ThreadedPipeline pipeline(
        source, *runner, edge_ai::PostprocessOptions{}, pipeline_options);
    edge_ai::BatchResult batch;
    edge_ai::PipelineStats stats;
    std::string fatal_error;
    const bool run_ok = pipeline.run(
        batch, stats, edge_ai::ResultCallback{}, fatal_error);
    if (!expect(!run_ok, "pipeline unexpectedly started all threads", error) ||
        !expect(fatal_error.find("failed to start " + stage_name + " thread") !=
                    std::string::npos,
                "thread-start failure lacks stage: " + fatal_error, error) ||
        !expect(fatal_error.find("injected thread start failure") !=
                    std::string::npos,
                "thread-start failure lacks injected detail: " + fatal_error,
                error)) {
        return false;
    }
    return true;
}

bool testStartFailureInference(std::string& error)
{
    return runStartFailure(edge_ai::PipelineStage::Inference, "inference",
                           error);
}

bool testStartFailureResult(std::string& error)
{
    return runStartFailure(edge_ai::PipelineStage::Result, "result", error);
}

}  // namespace

int main(int argc, char* argv[])
{
    const TestCase tests[] = {
        {"threaded_normal_drain", testNormalDrain},
        {"threaded_full_backpressure", testFullBackpressure},
        {"threaded_fatal_wakes", testFatalWakesWaiters},
        {"threaded_start_failure_inference", testStartFailureInference},
        {"threaded_start_failure_result", testStartFailureResult},
    };

    const std::string filter = argc > 1 ? argv[1] : "";
    int ran = 0;
    int failures = 0;
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
        std::cerr << "error: unknown threaded pipeline test: " << filter
                  << "\n";
        return 2;
    }
    std::cout << ran << " test(s), " << failures << " failure(s)\n";
    return failures == 0 ? 0 : 1;
}

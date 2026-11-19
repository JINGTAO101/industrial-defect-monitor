#include <atomic>
#include <chrono>
#include <condition_variable>
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

bool wait_gate_ready(
    std::size_t input_depth,
    bool input_waiting,
    std::size_t result_depth,
    bool result_waiting,
    std::size_t capacity)
{
    // Both queues must be full AND both producers must currently be blocked.
    // A stale wait flag with an available slot must not open the gate.
    return input_depth == capacity && input_waiting &&
           result_depth == capacity && result_waiting;
}

struct ProducerWaitState {
    std::mutex mutex;
    std::condition_variable condition;
    std::size_t capacity = 1;
    std::size_t input_depth = 0;
    std::size_t result_depth = 0;
    bool input_waiting = false;
    bool result_waiting = false;

    void set_input_depth(std::size_t depth)
    {
        std::lock_guard<std::mutex> lock(mutex);
        input_depth = depth;
        condition.notify_all();
    }

    void set_result_depth(std::size_t depth)
    {
        std::lock_guard<std::mutex> lock(mutex);
        result_depth = depth;
        condition.notify_all();
    }

    void set_input_waiting(bool waiting)
    {
        std::lock_guard<std::mutex> lock(mutex);
        input_waiting = waiting;
        condition.notify_all();
    }

    void set_result_waiting(bool waiting)
    {
        std::lock_guard<std::mutex> lock(mutex);
        result_waiting = waiting;
        condition.notify_all();
    }

    std::function<void()> input_start()
    {
        return [this] { set_input_waiting(true); };
    }

    std::function<void()> input_end()
    {
        return [this] { set_input_waiting(false); };
    }

    std::function<void()> result_start()
    {
        return [this] { set_result_waiting(true); };
    }

    std::function<void()> result_end()
    {
        return [this] { set_result_waiting(false); };
    }

    std::function<void(std::size_t)> input_depth_hook()
    {
        return [this](std::size_t depth) { set_input_depth(depth); };
    }

    std::function<void(std::size_t)> result_depth_hook()
    {
        return [this](std::size_t depth) { set_result_depth(depth); };
    }

    bool ready()
    {
        std::lock_guard<std::mutex> lock(mutex);
        return wait_gate_ready(input_depth, input_waiting, result_depth,
                               result_waiting, capacity);
    }

    bool wait_both(std::chrono::milliseconds timeout)
    {
        std::unique_lock<std::mutex> lock(mutex);
        return condition.wait_for(lock, timeout, [this] {
            return wait_gate_ready(input_depth, input_waiting, result_depth,
                                   result_waiting, capacity);
        });
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
    ProducerWaitState wait_state;
    wait_state.capacity = 1;
    edge_ai::PipelineOptions pipeline_options;
    pipeline_options.input_queue_capacity = 1;
    pipeline_options.result_queue_capacity = 1;
    pipeline_options.input_producer_wait_hook = wait_state.input_start();
    pipeline_options.input_producer_wake_hook = wait_state.input_end();
    pipeline_options.result_producer_wait_hook = wait_state.result_start();
    pipeline_options.result_producer_wake_hook = wait_state.result_end();
    pipeline_options.input_depth_hook = wait_state.input_depth_hook();
    pipeline_options.result_depth_hook = wait_state.result_depth_hook();
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
        !wait_state.wait_both(2000ms)) {
        timed_out = true;
        release_result.set_value();
    } else {
        release_result.set_value();
    }
    runner_thread.join();

    if (timed_out) {
        return expect(false,
                      "pipeline did not have both producers waiting before release",
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
    ProducerWaitState wait_state;
    wait_state.capacity = 1;
    edge_ai::PipelineOptions pipeline_options;
    pipeline_options.input_queue_capacity = 1;
    pipeline_options.result_queue_capacity = 1;
    pipeline_options.input_producer_wait_hook = wait_state.input_start();
    pipeline_options.input_producer_wake_hook = wait_state.input_end();
    pipeline_options.result_producer_wait_hook = wait_state.result_start();
    pipeline_options.result_producer_wake_hook = wait_state.result_end();
    pipeline_options.input_depth_hook = wait_state.input_depth_hook();
    pipeline_options.result_depth_hook = wait_state.result_depth_hook();
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
        !wait_state.wait_both(2000ms)) {
        timed_out = true;
        fail_result.set_value();
    } else {
        fail_result.set_value();
    }
    runner_thread.join();

    if (timed_out) {
        return expect(false,
                      "pipeline did not have both producers waiting before the fatal error",
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

bool testWaitGateRejectsStaleFlag(std::string& error)
{
    ProducerWaitState state;
    state.capacity = 1;
    // Input queue has space, but the old wait flag has not been cleared yet.
    state.set_input_depth(0);
    state.set_input_waiting(true);
    state.set_result_depth(1);
    state.set_result_waiting(true);
    if (!expect(!state.ready(),
                "wait gate opened with an available input slot", error)) {
        return false;
    }

    // Only when both queues are full and both producers wait may the gate open.
    state.set_input_depth(1);
    return expect(state.ready(),
                  "wait gate did not open with both queues full and waiting",
                  error);
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
        {"threaded_wait_gate_stale_flag", testWaitGateRejectsStaleFlag},
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

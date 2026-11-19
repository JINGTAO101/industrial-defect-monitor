#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "annotation.hpp"

namespace edge_ai {

enum class PreviewStatus { Ok, Closed, Error };

// Display backend seam: production uses OpenCV highgui, tests use a fake.
class PreviewSink {
public:
    virtual ~PreviewSink() = default;
    // Returns Closed when the user closed the window (a normal stop), or
    // Error with a diagnostic when the backend cannot display.
    virtual PreviewStatus show(const cv::Mat& annotated,
                               std::string& error) = 0;
    virtual void close() noexcept {}
};

std::unique_ptr<PreviewSink> makeOpenCvPreviewSink(std::string title);

// Blocking single-image preview. Returns false only on a display error;
// closing the window is a normal end.
bool showAnnotatedWindow(const cv::Mat& image,
                         const AnnotationInput& input,
                         const std::string& title,
                         std::string& error);

struct PreviewStats {
    std::size_t submitted = 0;
    std::size_t shown = 0;
    std::size_t dropped = 0;
    bool closed_by_user = false;
    bool failed = false;
    std::string failure;
};

// Bounded preview: only the newest pending frame is kept. submit() never
// blocks on the window, so a slow display drops preview frames without
// delaying detection results. Closing the window sets external_stop so the
// camera pipeline drains exactly like 5B Ctrl+C.
class PreviewWindow {
public:
    PreviewWindow(std::string title,
                  std::atomic<bool>* external_stop);
    // Test seam: sink supplies the display behavior.
    PreviewWindow(std::string title,
                  std::atomic<bool>* external_stop,
                  std::unique_ptr<PreviewSink> sink);
    ~PreviewWindow();

    PreviewWindow(const PreviewWindow&) = delete;
    PreviewWindow& operator=(const PreviewWindow&) = delete;

    bool start(std::string& error);
    void submit(cv::Mat image, AnnotationInput input);
    void close();

    PreviewStats stats() const;
    bool failed() const;
    std::string failure() const;

private:
    struct Pending {
        cv::Mat image;
        AnnotationInput input;
    };

    void run();
    void requestStop();
    void closeSink();

    std::string title_;
    std::atomic<bool>* external_stop_ = nullptr;
    std::unique_ptr<PreviewSink> sink_;

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::unique_ptr<Pending> pending_;
    bool stopping_ = false;
    bool finished_ = false;
    bool closed_by_user_ = false;
    bool failed_ = false;
    std::string failure_;
    std::size_t submitted_ = 0;
    std::size_t shown_ = 0;
    std::size_t dropped_ = 0;

    std::thread thread_;
    std::atomic<bool> sink_closed_{false};
};

}  // namespace edge_ai

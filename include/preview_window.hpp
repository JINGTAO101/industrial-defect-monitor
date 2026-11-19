#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>

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
    // Processes window events when no new frame is available. This is what
    // lets a close request be noticed while the camera is disconnected.
    virtual PreviewStatus poll(std::string& error)
    {
        (void)error;
        return PreviewStatus::Ok;
    }
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

// Main-thread driven preview. All highgui calls happen from pump()/close(),
// which must run on the thread that owns the window (the process main thread
// for the CLI). submit() is called by the result thread and never blocks on
// the window: at most one pending frame is kept, so a slow display skips
// previews without delaying detection results.
//
// pump() also polls the sink when no frame is pending, so closing the window
// still triggers the 5B external stop while the camera has no frames.
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
    void pump();
    void close();

    PreviewStats stats() const;
    bool failed() const;
    bool finished() const;
    std::string failure() const;

private:
    struct Pending {
        cv::Mat image;
        AnnotationInput input;
    };

    void finish(PreviewStatus status, const std::string& error);
    void requestStop();
    void closeSink();

    std::string title_;
    std::atomic<bool>* external_stop_ = nullptr;
    std::unique_ptr<PreviewSink> sink_;

    mutable std::mutex mutex_;
    std::unique_ptr<Pending> pending_;
    bool started_ = false;
    bool stopping_ = false;
    bool finished_ = false;
    bool closed_by_user_ = false;
    bool failed_ = false;
    std::string failure_;
    std::size_t submitted_ = 0;
    std::size_t shown_ = 0;
    std::size_t dropped_ = 0;
    std::atomic<bool> sink_closed_{false};
};

}  // namespace edge_ai

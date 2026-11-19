#include "preview_window.hpp"

#include <exception>
#include <utility>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace edge_ai {
namespace {

class OpenCvPreviewSink final : public PreviewSink {
public:
    explicit OpenCvPreviewSink(std::string title) : title_(std::move(title)) {}

    PreviewStatus show(const cv::Mat& annotated,
                       std::string& error) override
    {
        try {
            if (!window_created_) {
                cv::namedWindow(title_, cv::WINDOW_NORMAL);
                window_created_ = true;
            }
            cv::imshow(title_, annotated);
            return processEvents(error);
        } catch (const cv::Exception& exception) {
            error = std::string("preview display failed: ") + exception.what();
            return PreviewStatus::Error;
        } catch (const std::exception& exception) {
            error = std::string("preview display failed: ") + exception.what();
            return PreviewStatus::Error;
        }
    }

    PreviewStatus poll(std::string& error) override
    {
        if (!window_created_) {
            return PreviewStatus::Ok;
        }
        try {
            return processEvents(error);
        } catch (const cv::Exception& exception) {
            error = std::string("preview display failed: ") + exception.what();
            return PreviewStatus::Error;
        } catch (const std::exception& exception) {
            error = std::string("preview display failed: ") + exception.what();
            return PreviewStatus::Error;
        }
    }

    void close() noexcept override
    {
        if (!window_created_) {
            return;
        }
        window_created_ = false;
        try {
            cv::destroyWindow(title_);
        } catch (...) {
            // A failed display backend must not throw from teardown.
        }
    }

private:
    PreviewStatus processEvents(std::string& error)
    {
        // waitKey(1) pumps the Qt event queue even when no new image arrives.
        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            return PreviewStatus::Closed;
        }
        if (cv::getWindowProperty(title_, cv::WND_PROP_VISIBLE) < 1.0) {
            return PreviewStatus::Closed;
        }
        (void)error;
        return PreviewStatus::Ok;
    }

    std::string title_;
    bool window_created_ = false;
};

}  // namespace

std::unique_ptr<PreviewSink> makeOpenCvPreviewSink(std::string title)
{
    return std::make_unique<OpenCvPreviewSink>(std::move(title));
}

bool showAnnotatedWindow(const cv::Mat& image,
                         const AnnotationInput& input,
                         const std::string& title,
                         std::string& error)
{
    try {
        const cv::Mat annotated = annotateFrame(image, input);
        if (annotated.empty()) {
            error = "cannot display an empty annotated frame";
            return false;
        }
        cv::namedWindow(title, cv::WINDOW_NORMAL);
        cv::imshow(title, annotated);
        // Any key or closing the window ends the single-image preview.
        while (true) {
            const int key = cv::waitKey(30);
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
            if (cv::getWindowProperty(title, cv::WND_PROP_VISIBLE) < 1.0) {
                break;
            }
        }
        cv::destroyWindow(title);
        return true;
    } catch (const cv::Exception& exception) {
        error = std::string("preview display failed: ") + exception.what();
        return false;
    } catch (const std::exception& exception) {
        error = std::string("preview display failed: ") + exception.what();
        return false;
    }
}

PreviewWindow::PreviewWindow(std::string title,
                             std::atomic<bool>* external_stop)
    : PreviewWindow(std::move(title), external_stop,
                    makeOpenCvPreviewSink("edge_ai preview"))
{
}

PreviewWindow::PreviewWindow(std::string title,
                             std::atomic<bool>* external_stop,
                             std::unique_ptr<PreviewSink> sink)
    : title_(std::move(title)),
      external_stop_(external_stop),
      sink_(std::move(sink))
{
}

PreviewWindow::~PreviewWindow()
{
    close();
}

bool PreviewWindow::start(std::string& error)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (started_) {
        error = "preview window is already started";
        return false;
    }
    if (!sink_) {
        error = "preview sink is null";
        return false;
    }
    started_ = true;
    return true;
}

void PreviewWindow::submit(cv::Mat image, AnnotationInput input)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!started_ || stopping_ || finished_) {
        return;
    }
    if (pending_) {
        // Capacity is exactly one frame: keep the newest preview and count
        // the skipped one. Detection processing is unaffected.
        ++dropped_;
    }
    pending_ =
        std::make_unique<Pending>(Pending{std::move(image), std::move(input)});
    ++submitted_;
}

void PreviewWindow::pump()
{
    std::unique_ptr<Pending> pending;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!started_ || stopping_ || finished_) {
            return;
        }
        if (pending_) {
            pending = std::move(pending_);
        }
    }

    std::string error;
    PreviewStatus status = PreviewStatus::Ok;
    bool had_frame = pending != nullptr;
    try {
        if (had_frame) {
            const cv::Mat annotated =
                annotateFrame(pending->image, pending->input);
            if (annotated.empty()) {
                finish(PreviewStatus::Error,
                       "preview frame is empty or has an unsupported format");
                return;
            }
            status = sink_->show(annotated, error);
        } else {
            status = sink_->poll(error);
        }
    } catch (const std::exception& exception) {
        status = PreviewStatus::Error;
        error = std::string("preview display failed: ") + exception.what();
    } catch (...) {
        status = PreviewStatus::Error;
        error = "preview display failed: unknown exception";
    }

    if (status == PreviewStatus::Ok) {
        if (had_frame) {
            std::lock_guard<std::mutex> lock(mutex_);
            ++shown_;
        }
        return;
    }
    finish(status, error);
}

void PreviewWindow::close()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        pending_.reset();
    }
    closeSink();
}

PreviewStats PreviewWindow::stats() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    PreviewStats result;
    result.submitted = submitted_;
    result.shown = shown_;
    result.dropped = dropped_;
    result.closed_by_user = closed_by_user_;
    result.failed = failed_;
    result.failure = failure_;
    return result;
}

bool PreviewWindow::failed() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return failed_;
}

bool PreviewWindow::finished() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return finished_ || stopping_;
}

std::string PreviewWindow::failure() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return failure_;
}

void PreviewWindow::finish(PreviewStatus status, const std::string& error)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        if (status == PreviewStatus::Closed) {
            closed_by_user_ = true;
        } else {
            failed_ = true;
            failure_ = error.empty() ? "preview display failed" : error;
        }
    }
    requestStop();
}

void PreviewWindow::requestStop()
{
    if (external_stop_ != nullptr) {
        external_stop_->store(true);
    }
}

void PreviewWindow::closeSink()
{
    if (!sink_) {
        return;
    }
    bool expected = false;
    if (sink_closed_.compare_exchange_strong(expected, true)) {
        sink_->close();
    }
}

}  // namespace edge_ai

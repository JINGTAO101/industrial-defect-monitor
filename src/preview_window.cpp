#include "preview_window.hpp"

#include <algorithm>
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
            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                return PreviewStatus::Closed;
            }
            if (cv::getWindowProperty(title_, cv::WND_PROP_VISIBLE) < 1.0) {
                return PreviewStatus::Closed;
            }
            return PreviewStatus::Ok;
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
    if (thread_.joinable()) {
        error = "preview window is already started";
        return false;
    }
    if (!sink_) {
        error = "preview sink is null";
        return false;
    }
    try {
        thread_ = std::thread([this] { run(); });
    } catch (const std::exception& exception) {
        error = std::string("failed to start preview thread: ") +
                exception.what();
        return false;
    }
    return true;
}

void PreviewWindow::submit(cv::Mat image, AnnotationInput input)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || finished_) {
            return;
        }
        if (pending_) {
            // Capacity is exactly one frame: keep the newest preview and
            // count the skipped one. Detection processing is unaffected.
            ++dropped_;
        }
        pending_ = std::make_unique<Pending>(
            Pending{std::move(image), std::move(input)});
        ++submitted_;
    }
    condition_.notify_one();
}

void PreviewWindow::close()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        pending_.reset();
    }
    condition_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
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

std::string PreviewWindow::failure() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return failure_;
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
        // Run on the preview thread whenever it exists: highgui window
        // creation and destruction should stay on one thread.
        sink_->close();
    }
}

void PreviewWindow::run()
{
    while (true) {
        Pending pending;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] {
                return stopping_ || pending_ != nullptr;
            });
            if (stopping_) {
                break;
            }
            pending = std::move(*pending_);
            pending_.reset();
        }

        const cv::Mat annotated =
            annotateFrame(pending.image, pending.input);
        if (annotated.empty()) {
            std::lock_guard<std::mutex> lock(mutex_);
            failed_ = true;
            failure_ = "preview frame is empty or has an unsupported format";
            finished_ = true;
            requestStop();
            break;
        }

        std::string show_error;
        PreviewStatus status = PreviewStatus::Error;
        try {
            status = sink_->show(annotated, show_error);
        } catch (const std::exception& exception) {
            show_error = std::string("preview display failed: ") +
                         exception.what();
        } catch (...) {
            show_error = "preview display failed: unknown exception";
        }

        if (status == PreviewStatus::Ok) {
            std::lock_guard<std::mutex> lock(mutex_);
            ++shown_;
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            finished_ = true;
            if (status == PreviewStatus::Closed) {
                closed_by_user_ = true;
            } else {
                failed_ = true;
                failure_ = show_error.empty() ? "preview display failed"
                                             : show_error;
            }
        }
        requestStop();
        break;
    }
    closeSink();
}

}  // namespace edge_ai

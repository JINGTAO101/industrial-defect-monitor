#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "image_source.hpp"

namespace edge_ai {

struct CameraOptions {
    std::string device = "/dev/video0";
    std::string format = "YUYV";
    int width = 640;
    int height = 480;
    int fps = 30;
    int timeout_ms = 2000;
};

struct CameraInfo {
    std::string device;
    std::string requested_format;
    int requested_width = 0;
    int requested_height = 0;
    int requested_fps = 0;
    std::string actual_format;
    int actual_width = 0;
    int actual_height = 0;
    double actual_fps = 0.0;  // <= 0 means the driver value is unknown
    int bytes_per_line = 0;
    std::string colorspace;
    std::string ycbcr_encoding;
    std::string quantization;
};

// Initialization ioctl errors: EBUSY is always surfaced as "device busy",
// permission errors distinctly, and other errors keep the caller's context.
std::string cameraInitializationError(
    const std::string& context,
    const std::string& device,
    int error_number);

// Accepts only YUYV with BT.601/default chroma encoding and limited/default
// quantization; all other colorimetry is rejected rather than mis-converted.
bool supportedYuyvColorimetry(
    std::uint32_t colorspace,
    std::uint32_t ycbcr_encoding,
    std::uint32_t quantization);

// Validates a dequeued buffer before copying it. Error-flagged, empty and
// overlength frames are rejected; the caller still requeues valid indices.
bool validateCameraBuffer(
    bool error_flag,
    std::size_t index,
    std::size_t buffer_count,
    std::size_t bytesused,
    std::size_t buffer_length,
    std::string& error);

enum class CameraReadResult { FrameReady, Timeout, Error, Stopped };

struct CameraReadOps {
    // > 0 ready, 0 timeout, -1 error with error_number (EINTR is retried by
    // the helper using the remaining deadline, not a fresh full timeout).
    std::function<int(int timeout_ms, int& error_number)> poll;
    // 0 success, -1 error with error_number (EAGAIN/EINTR retried until the
    // shared deadline expires).
    std::function<int(int& error_number)> dequeue;
    // Optional external stop (Ctrl+C). When set, poll is sliced so a blocked
    // read observes the flag within ~50ms instead of the full timeout.
    std::function<bool()> should_stop;
};

CameraReadResult waitForCameraFrame(
    const CameraReadOps& operations,
    int timeout_ms,
    std::string& error);

// Performs exactly one VIDIOC_DQBUF syscall through ioctl_fn. Production and
// regression tests use this same one-shot helper; retry decisions belong to
// waitForCameraFrame()'s deadline loop.
int dequeueCameraBufferOnce(
    int fd,
    unsigned long request,
    void* buffer,
    const std::function<int(int, unsigned long, void*)>& ioctl_fn);

enum class CameraErrorKind { None, BadFrame, Timeout, DeviceLost, Fatal };

std::string normalizeCameraFormat(const std::string& format);
bool isSupportedCameraFormat(const std::string& format);

// Test seam: nextFrame returns a complete payload copied out of the driver
// buffer; for V4L2 the buffer has already been requeued before returning.
class CameraBackend {
public:
    virtual ~CameraBackend() = default;
    virtual bool nextFrame(std::vector<std::uint8_t>& payload,
                           std::string& error) = 0;
    virtual const CameraInfo& info() const = 0;
    virtual CameraErrorKind lastErrorKind() const { return CameraErrorKind::None; }
    // Optional stop propagation for backends that can block on a frame read.
    virtual void setExternalStop(const std::atomic<bool>*) {}
    virtual void close() noexcept = 0;
};

class CameraSource final : public ImageSource {
public:
    CameraSource(const CameraOptions& options, std::string& error);

    CameraSource(const CameraOptions& options,
                 std::unique_ptr<CameraBackend> backend,
                 std::string& error);

    ~CameraSource() override;

    SourceStatus next(Frame& frame, std::string& error) override;

    // Lets a blocked read return End promptly when Ctrl+C is requested.
    void setExternalStop(const std::atomic<bool>* external_stop);

    const CameraInfo& info() const;
    CameraErrorKind lastErrorKind() const;

private:
    bool shouldStop() const;

    CameraOptions options_;
    CameraInfo info_;
    std::unique_ptr<CameraBackend> backend_;
    const std::atomic<bool>* external_stop_ = nullptr;
    std::size_t next_index_ = 0;
    CameraErrorKind last_error_kind_ = CameraErrorKind::None;
};

using CameraFactory =
    std::function<std::unique_ptr<CameraSource>(const CameraOptions&, std::string&)>;
using CameraLogCallback = std::function<void(const std::string&)>;

// Wraps a V4L2 CameraSource and reopens it after a device-loss/timeout.
// Bad frames are returned as DataError; reconnect exhaustion is Fatal.
class ReconnectingCameraSource final : public ImageSource {
public:
    ReconnectingCameraSource(
        const CameraOptions& options,
        int max_attempts,
        int reconnect_interval_ms,
        const std::atomic<bool>* external_stop,
        std::string& error,
        CameraLogCallback log = {});

    // Test seam: factory creates each CameraSource instance.
    ReconnectingCameraSource(
        const CameraOptions& options,
        int max_attempts,
        int reconnect_interval_ms,
        const std::atomic<bool>* external_stop,
        CameraFactory factory,
        std::string& error,
        CameraLogCallback log = {});

    SourceStatus next(Frame& frame, std::string& error) override;
    SourceErrorKind errorKind() const override;
    const CameraInfo& info() const;
    std::size_t reconnectCount() const;

private:
    enum class ReconnectResult { Success, Stopped, Exhausted };

    bool shouldStop() const;
    void log(const std::string& message) const;
    ReconnectResult reconnect(std::string& error);

    CameraOptions options_;
    int max_attempts_ = 0;
    int reconnect_interval_ms_ = 0;
    const std::atomic<bool>* external_stop_ = nullptr;
    CameraFactory factory_;
    CameraLogCallback log_;
    std::unique_ptr<CameraSource> camera_;
    CameraInfo info_;
    std::size_t next_index_ = 0;
    std::size_t reconnect_count_ = 0;
    // Reopen budget for the current outage. Reset only after a valid frame so
    // repeated "reopen succeeds but reads keep timing out" cannot loop forever.
    int outage_attempts_ = 0;
    bool outage_active_ = false;
    SourceErrorKind error_kind_ = SourceErrorKind::DataError;
};

}  // namespace edge_ai

#pragma once

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

enum class CameraReadResult { FrameReady, Timeout, Error };

struct CameraReadOps {
    // > 0 ready, 0 timeout, -1 error with error_number (EINTR is retried by
    // the helper using the remaining deadline, not a fresh full timeout).
    std::function<int(int timeout_ms, int& error_number)> poll;
    // 0 success, -1 error with error_number (EAGAIN/EINTR retried until the
    // shared deadline expires).
    std::function<int(int& error_number)> dequeue;
};

CameraReadResult waitForCameraFrame(
    const CameraReadOps& operations,
    int timeout_ms,
    std::string& error);

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

    const CameraInfo& info() const;

private:
    CameraOptions options_;
    CameraInfo info_;
    std::unique_ptr<CameraBackend> backend_;
    std::size_t next_index_ = 0;
};

}  // namespace edge_ai

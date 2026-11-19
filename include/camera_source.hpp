#pragma once

#include <cstdint>
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
    double actual_fps = 0.0;
    int bytes_per_line = 0;
};

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

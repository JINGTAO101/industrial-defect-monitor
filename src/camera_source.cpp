#include "camera_source.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include <linux/videodev2.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace edge_ai {
namespace {

constexpr std::size_t kBufferCount = 4;

std::string errnoMessage(int error_number)
{
    return std::strerror(error_number);
}

std::string fourccToString(std::uint32_t fourcc)
{
    char text[5] = {
        static_cast<char>(fourcc & 0xff),
        static_cast<char>((fourcc >> 8) & 0xff),
        static_cast<char>((fourcc >> 16) & 0xff),
        static_cast<char>((fourcc >> 24) & 0xff),
        '\0'};
    return text;
}

bool validateOptions(
    const CameraOptions& options,
    CameraInfo& info,
    std::string& error)
{
    if (options.device.empty()) {
        error = "camera device path is empty";
        return false;
    }
    const std::string format = normalizeCameraFormat(options.format);
    if (format.empty()) {
        error = "unsupported camera pixel format: '" + options.format +
                "' (supported: MJPG, YUYV)";
        return false;
    }
    if (options.width <= 0 || options.height <= 0) {
        error = "camera width and height must be greater than 0";
        return false;
    }
    if (options.fps <= 0) {
        error = "camera fps must be greater than 0";
        return false;
    }
    if (options.timeout_ms <= 0) {
        error = "camera timeout must be greater than 0 ms";
        return false;
    }

    info.device = options.device;
    info.requested_format = format;
    info.requested_width = options.width;
    info.requested_height = options.height;
    info.requested_fps = options.fps;
    info.actual_format = format;
    info.actual_width = options.width;
    info.actual_height = options.height;
    info.actual_fps = static_cast<double>(options.fps);
    info.bytes_per_line = 0;
    return true;
}

bool decodePayload(
    const std::vector<std::uint8_t>& payload,
    const CameraInfo& info,
    cv::Mat& image,
    std::string& error)
{
    try {
        if (info.actual_format == "MJPG") {
            image = cv::imdecode(payload, cv::IMREAD_COLOR);
        } else if (info.actual_format == "YUYV") {
            const int stride =
                info.bytes_per_line > 0 ? info.bytes_per_line
                                        : info.actual_width * 2;
            const std::size_t required =
                static_cast<std::size_t>(stride) * info.actual_height;
            if (payload.size() < required) {
                error = "camera YUYV frame is too short: got " +
                        std::to_string(payload.size()) + " bytes, need " +
                        std::to_string(required);
                return false;
            }
            cv::Mat yuyv(
                info.actual_height, info.actual_width, CV_8UC2,
                const_cast<std::uint8_t*>(payload.data()), stride);
            cv::cvtColor(yuyv, image, cv::COLOR_YUV2BGR_YUYV);
        } else {
            error = "unsupported camera frame format: " + info.actual_format;
            return false;
        }
    } catch (const cv::Exception& exception) {
        error = std::string("failed to decode camera frame: ") +
                exception.what();
        return false;
    }

    if (image.empty()) {
        error = "failed to decode camera frame: empty image";
        return false;
    }
    if (image.type() != CV_8UC3) {
        error = "camera frame did not decode to CV_8UC3 BGR";
        return false;
    }
    return true;
}

class V4L2Backend final : public CameraBackend {
public:
    ~V4L2Backend() override
    {
        close();
    }

    bool start(const CameraOptions& options, std::string& error)
    {
        close();
        error.clear();

        fd_ = ::open(options.device.c_str(), O_RDWR | O_NONBLOCK, 0);
        if (fd_ < 0) {
            const int saved_errno = errno;
            if (saved_errno == ENOENT || saved_errno == ENODEV) {
                error = "camera device not found: " + options.device + ": " +
                        errnoMessage(saved_errno);
            } else if (saved_errno == EBUSY) {
                error = "camera device busy: " + options.device + ": " +
                        errnoMessage(saved_errno);
            } else if (saved_errno == EACCES || saved_errno == EPERM) {
                error = "camera permission denied: " + options.device + ": " +
                        errnoMessage(saved_errno);
            } else {
                error = "failed to open camera " + options.device + ": " +
                        errnoMessage(saved_errno);
            }
            return false;
        }

        v4l2_capability capability{};
        if (xioctl(VIDIOC_QUERYCAP, &capability) < 0) {
            error = "camera ioctl VIDIOC_QUERYCAP failed: " +
                    errnoMessage(errno);
            close();
            return false;
        }
        if ((capability.capabilities & V4L2_CAP_VIDEO_CAPTURE) == 0) {
            error = "camera does not support video capture: " + options.device;
            close();
            return false;
        }
        if ((capability.capabilities & V4L2_CAP_STREAMING) == 0) {
            error = "camera does not support streaming: " + options.device;
            close();
            return false;
        }

        const std::string requested_format =
            normalizeCameraFormat(options.format);
        std::uint32_t fourcc = 0;
        if (requested_format == "MJPG") {
            fourcc = V4L2_PIX_FMT_MJPEG;
        } else if (requested_format == "YUYV") {
            fourcc = V4L2_PIX_FMT_YUYV;
        } else {
            error = "unsupported camera pixel format: " + options.format;
            close();
            return false;
        }

        v4l2_format format{};
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        format.fmt.pix.width = static_cast<std::uint32_t>(options.width);
        format.fmt.pix.height = static_cast<std::uint32_t>(options.height);
        format.fmt.pix.pixelformat = fourcc;
        format.fmt.pix.field = V4L2_FIELD_ANY;
        if (xioctl(VIDIOC_S_FMT, &format) < 0) {
            error = "camera format not supported: requested " +
                    requested_format + " " + std::to_string(options.width) +
                    "x" + std::to_string(options.height) + ": " +
                    errnoMessage(errno);
            close();
            return false;
        }
        if (format.fmt.pix.pixelformat != fourcc) {
            error = "camera format not supported: requested " +
                    requested_format + ", driver selected " +
                    fourccToString(format.fmt.pix.pixelformat);
            close();
            return false;
        }

        info_.device = options.device;
        info_.requested_format = requested_format;
        info_.requested_width = options.width;
        info_.requested_height = options.height;
        info_.requested_fps = options.fps;
        info_.actual_format = requested_format;
        info_.actual_width = static_cast<int>(format.fmt.pix.width);
        info_.actual_height = static_cast<int>(format.fmt.pix.height);
        info_.bytes_per_line = static_cast<int>(format.fmt.pix.bytesperline);
        info_.actual_fps = static_cast<double>(options.fps);

        v4l2_streamparm parameters{};
        parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parameters.parm.capture.timeperframe.numerator = 1;
        parameters.parm.capture.timeperframe.denominator =
            static_cast<std::uint32_t>(options.fps);
        if (xioctl(VIDIOC_S_PARM, &parameters) < 0) {
            error = "camera frame rate not supported: requested " +
                    std::to_string(options.fps) + " fps: " +
                    errnoMessage(errno);
            close();
            return false;
        }
        v4l2_streamparm actual_parameters{};
        actual_parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(VIDIOC_G_PARM, &actual_parameters) == 0) {
            const auto& timeperframe =
                actual_parameters.parm.capture.timeperframe;
            if (timeperframe.numerator != 0) {
                info_.actual_fps =
                    static_cast<double>(timeperframe.denominator) /
                    static_cast<double>(timeperframe.numerator);
            }
        }

        v4l2_requestbuffers request{};
        request.count = static_cast<std::uint32_t>(kBufferCount);
        request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        request.memory = V4L2_MEMORY_MMAP;
        if (xioctl(VIDIOC_REQBUFS, &request) < 0) {
            error = "camera buffer setup failed: " + errnoMessage(errno);
            close();
            return false;
        }
        if (request.count == 0) {
            error = "camera driver allocated no capture buffers";
            close();
            return false;
        }

        buffers_.resize(request.count);
        for (std::uint32_t index = 0; index < request.count; ++index) {
            v4l2_buffer buffer{};
            buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buffer.memory = V4L2_MEMORY_MMAP;
            buffer.index = index;
            if (xioctl(VIDIOC_QUERYBUF, &buffer) < 0) {
                error = "camera buffer query failed: " + errnoMessage(errno);
                close();
                return false;
            }
            void* mapped = mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd_, buffer.m.offset);
            if (mapped == MAP_FAILED) {
                error = "camera buffer mmap failed: " + errnoMessage(errno);
                close();
                return false;
            }
            buffers_[index].start = mapped;
            buffers_[index].length = buffer.length;

            v4l2_buffer queued{};
            queued.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            queued.memory = V4L2_MEMORY_MMAP;
            queued.index = index;
            if (xioctl(VIDIOC_QBUF, &queued) < 0) {
                error = "camera buffer queue failed: " + errnoMessage(errno);
                close();
                return false;
            }
        }

        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(VIDIOC_STREAMON, &type) < 0) {
            error = "camera stream start failed: " + errnoMessage(errno);
            close();
            return false;
        }
        streaming_ = true;
        return true;
    }

    bool nextFrame(std::vector<std::uint8_t>& payload,
                   std::string& error) override
    {
        error.clear();
        if (fd_ < 0 || !streaming_) {
            error = "camera is not streaming";
            return false;
        }

        pollfd descriptor{fd_, POLLIN, 0};
        int poll_result = 0;
        do {
            poll_result = poll(&descriptor, 1, timeout_ms_);
        } while (poll_result < 0 && errno == EINTR);
        if (poll_result == 0) {
            error = "camera read timeout after " +
                    std::to_string(timeout_ms_) + " ms";
            return false;
        }
        if (poll_result < 0) {
            error = "camera poll failed: " + errnoMessage(errno);
            return false;
        }
        if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
            error = "camera device error or hangup";
            return false;
        }

        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        if (xioctl(VIDIOC_DQBUF, &buffer) < 0) {
            if (errno == EAGAIN) {
                error = "camera read timeout after " +
                        std::to_string(timeout_ms_) + " ms";
            } else {
                error = "camera dequeue failed: " + errnoMessage(errno);
            }
            return false;
        }
        if (buffer.index >= buffers_.size()) {
            error = "camera returned an invalid buffer index";
            return false;
        }
        if (buffer.bytesused == 0 ||
            buffer.bytesused > buffers_[buffer.index].length) {
            error = "camera returned an invalid frame size";
            return false;
        }

        const auto* data =
            static_cast<const std::uint8_t*>(buffers_[buffer.index].start);
        payload.assign(data, data + buffer.bytesused);

        v4l2_buffer requeue{};
        requeue.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        requeue.memory = V4L2_MEMORY_MMAP;
        requeue.index = buffer.index;
        if (xioctl(VIDIOC_QBUF, &requeue) < 0) {
            error = "failed to requeue camera buffer: " +
                    errnoMessage(errno);
            return false;
        }
        return true;
    }

    const CameraInfo& info() const override
    {
        return info_;
    }

    void close() noexcept override
    {
        if (fd_ >= 0 && streaming_) {
            v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            xioctl(VIDIOC_STREAMOFF, &type);
            streaming_ = false;
        }
        for (auto& buffer : buffers_) {
            if (buffer.start != nullptr && buffer.length > 0) {
                munmap(buffer.start, buffer.length);
            }
            buffer.start = nullptr;
            buffer.length = 0;
        }
        buffers_.clear();
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    void setTimeout(int timeout_ms)
    {
        timeout_ms_ = timeout_ms;
    }

private:
    struct Buffer {
        void* start = nullptr;
        std::size_t length = 0;
    };

    int xioctl(unsigned long request, void* argument)
    {
        int result = 0;
        do {
            result = ioctl(fd_, request, argument);
        } while (result < 0 && errno == EINTR);
        return result;
    }

    int fd_ = -1;
    bool streaming_ = false;
    int timeout_ms_ = 2000;
    CameraInfo info_;
    std::vector<Buffer> buffers_;
};

}  // namespace

std::string normalizeCameraFormat(const std::string& format)
{
    std::string normalized = format;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::toupper(c));
                   });
    if (normalized == "MJPEG" || normalized == "MJPG") {
        return "MJPG";
    }
    if (normalized == "YUYV") {
        return "YUYV";
    }
    return {};
}

bool isSupportedCameraFormat(const std::string& format)
{
    return !normalizeCameraFormat(format).empty();
}

CameraSource::CameraSource(const CameraOptions& options, std::string& error)
    : options_(options)
{
    if (!validateOptions(options_, info_, error)) {
        return;
    }
    auto backend = std::make_unique<V4L2Backend>();
    backend->setTimeout(options_.timeout_ms);
    if (!backend->start(options_, error)) {
        return;
    }
    info_ = backend->info();
    backend_ = std::move(backend);
}

CameraSource::CameraSource(
    const CameraOptions& options,
    std::unique_ptr<CameraBackend> backend,
    std::string& error)
    : options_(options)
{
    if (!validateOptions(options_, info_, error)) {
        return;
    }
    if (!backend) {
        error = "camera backend is null";
        return;
    }
    info_ = backend->info();
    backend_ = std::move(backend);
}

CameraSource::~CameraSource()
{
    if (backend_) {
        backend_->close();
    }
}

SourceStatus CameraSource::next(Frame& frame, std::string& error)
{
    error.clear();
    if (!backend_) {
        error = "camera is not initialized";
        return SourceStatus::Error;
    }

    std::vector<std::uint8_t> payload;
    if (!backend_->nextFrame(payload, error)) {
        return SourceStatus::Error;
    }
    info_ = backend_->info();

    cv::Mat image;
    if (!decodePayload(payload, info_, image, error)) {
        return SourceStatus::Error;
    }

    frame.image = std::move(image);
    frame.source = info_.device;
    frame.index = next_index_++;
    frame.timestamp = std::chrono::system_clock::now();
    return SourceStatus::Frame;
}

const CameraInfo& CameraSource::info() const
{
    return info_;
}

}  // namespace edge_ai

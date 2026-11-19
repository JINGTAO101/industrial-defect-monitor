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
#include <system_error>
#include <sys/ioctl.h>
#include <thread>
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

std::string colorspaceToString(std::uint32_t colorspace)
{
    switch (colorspace) {
        case V4L2_COLORSPACE_DEFAULT:
            return "default";
        case V4L2_COLORSPACE_SRGB:
            return "sRGB";
        case V4L2_COLORSPACE_JPEG:
            return "JPEG";
        case V4L2_COLORSPACE_REC709:
            return "Rec.709";
        default:
            return "colorspace(" + std::to_string(colorspace) + ")";
    }
}

std::string ycbcrEncodingToString(std::uint32_t encoding)
{
    switch (encoding) {
        case V4L2_YCBCR_ENC_DEFAULT:
            return "default";
        case V4L2_YCBCR_ENC_601:
            return "BT.601";
        case V4L2_YCBCR_ENC_709:
            return "BT.709";
        case V4L2_YCBCR_ENC_XV601:
            return "xvYCC 601";
        case V4L2_YCBCR_ENC_XV709:
            return "xvYCC 709";
        default:
            return "encoding(" + std::to_string(encoding) + ")";
    }
}

std::string quantizationToString(std::uint32_t quantization)
{
    switch (quantization) {
        case V4L2_QUANTIZATION_DEFAULT:
            return "default";
        case V4L2_QUANTIZATION_FULL_RANGE:
            return "full";
        case V4L2_QUANTIZATION_LIM_RANGE:
            return "limited";
        default:
            return "quantization(" + std::to_string(quantization) + ")";
    }
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
            error = cameraInitializationError(
                "camera format not supported: requested " + requested_format +
                    " " + std::to_string(options.width) + "x" +
                    std::to_string(options.height),
                options.device, errno);
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
        if (fourcc == V4L2_PIX_FMT_YUYV &&
            !supportedYuyvColorimetry(format.fmt.pix.colorspace,
                                      format.fmt.pix.ycbcr_enc,
                                      format.fmt.pix.quantization)) {
            error = "camera YUYV colorimetry not supported: colorspace=" +
                    colorspaceToString(format.fmt.pix.colorspace) +
                    " ycbcr=" + ycbcrEncodingToString(format.fmt.pix.ycbcr_enc) +
                    " quantization=" +
                    quantizationToString(format.fmt.pix.quantization);
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
        info_.colorspace = colorspaceToString(format.fmt.pix.colorspace);
        info_.ycbcr_encoding =
            ycbcrEncodingToString(format.fmt.pix.ycbcr_enc);
        info_.quantization =
            quantizationToString(format.fmt.pix.quantization);
        info_.actual_fps = 0.0;

        v4l2_streamparm parameters{};
        parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parameters.parm.capture.timeperframe.numerator = 1;
        parameters.parm.capture.timeperframe.denominator =
            static_cast<std::uint32_t>(options.fps);
        if (xioctl(VIDIOC_S_PARM, &parameters) < 0) {
            error = cameraInitializationError(
                "camera frame rate not supported: requested " +
                    std::to_string(options.fps) + " fps",
                options.device, errno);
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
            error = cameraInitializationError("camera buffer setup failed",
                                              options.device, errno);
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
                error = cameraInitializationError("camera buffer query failed",
                                                  options.device, errno);
                close();
                return false;
            }
            void* mapped = mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd_, buffer.m.offset);
            if (mapped == MAP_FAILED) {
                error = cameraInitializationError("camera buffer mmap failed",
                                                  options.device, errno);
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
                error = cameraInitializationError("camera buffer queue failed",
                                                  options.device, errno);
                close();
                return false;
            }
        }

        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(VIDIOC_STREAMON, &type) < 0) {
            error = cameraInitializationError("camera stream start failed",
                                              options.device, errno);
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
        error_kind_ = CameraErrorKind::None;
        if (fd_ < 0 || !streaming_) {
            error_kind_ = CameraErrorKind::DeviceLost;
            error = "camera is not streaming";
            return false;
        }

        v4l2_buffer buffer{};
        CameraReadOps operations;
        operations.poll = [this](int timeout_ms, int& error_number) {
            pollfd descriptor{fd_, POLLIN, 0};
            const int result = poll(&descriptor, 1, timeout_ms);
            error_number = errno;
            if (result > 0 &&
                (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
                error_number = EIO;
                return -1;
            }
            return result;
        };
        operations.dequeue = [this, &buffer](int& error_number) {
            buffer = v4l2_buffer{};
            buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buffer.memory = V4L2_MEMORY_MMAP;
            // Single raw ioctl only: EINTR/EAGAIN must bubble up so the
            // shared deadline in waitForCameraFrame() decides whether to
            // retry. xioctl()'s internal retry would bypass the deadline.
            const int result = dequeueCameraBufferOnce(
                fd_, VIDIOC_DQBUF, &buffer,
                [](int fd, unsigned long request, void* argument) {
                    return ioctl(fd, request, argument);
                });
            error_number = errno;
            return result;
        };

        const CameraReadResult read_result =
            waitForCameraFrame(operations, timeout_ms_, error);
        if (read_result != CameraReadResult::FrameReady) {
            error_kind_ = read_result == CameraReadResult::Timeout
                              ? CameraErrorKind::Timeout
                              : CameraErrorKind::DeviceLost;
            return false;
        }

        const std::size_t index = buffer.index;
        const std::size_t buffer_length =
            index < buffers_.size() ? buffers_[index].length : 0;
        std::string buffer_error;
        const bool valid = validateCameraBuffer(
            (buffer.flags & V4L2_BUF_FLAG_ERROR) != 0, index,
            buffers_.size(), buffer.bytesused, buffer_length, buffer_error);

        if (valid) {
            const auto* data =
                static_cast<const std::uint8_t*>(buffers_[index].start);
            payload.assign(data, data + buffer.bytesused);
        } else {
            error_kind_ = index < buffers_.size()
                              ? CameraErrorKind::BadFrame
                              : CameraErrorKind::DeviceLost;
        }

        if (index < buffers_.size()) {
            v4l2_buffer requeue{};
            requeue.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            requeue.memory = V4L2_MEMORY_MMAP;
            requeue.index = buffer.index;
            if (xioctl(VIDIOC_QBUF, &requeue) < 0) {
                error = "failed to requeue camera buffer: " +
                        errnoMessage(errno);
                error_kind_ = CameraErrorKind::DeviceLost;
                return false;
            }
        }

        if (!valid) {
            error = buffer_error;
            return false;
        }
        return true;
    }

    const CameraInfo& info() const override
    {
        return info_;
    }

    CameraErrorKind lastErrorKind() const override
    {
        return error_kind_;
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
    CameraErrorKind error_kind_ = CameraErrorKind::None;
};

}  // namespace

std::string cameraInitializationError(
    const std::string& context,
    const std::string& device,
    int error_number)
{
    if (error_number == EBUSY) {
        return "camera device busy: " + device + ": " +
               errnoMessage(error_number);
    }
    if (error_number == EACCES || error_number == EPERM) {
        return "camera permission denied: " + device + ": " +
               errnoMessage(error_number);
    }
    return context + ": " + errnoMessage(error_number);
}

bool supportedYuyvColorimetry(
    std::uint32_t colorspace,
    std::uint32_t ycbcr_encoding,
    std::uint32_t quantization)
{
    const bool colorspace_ok =
        colorspace == V4L2_COLORSPACE_SRGB ||
        colorspace == V4L2_COLORSPACE_DEFAULT;
    const bool encoding_ok =
        ycbcr_encoding == V4L2_YCBCR_ENC_DEFAULT ||
        ycbcr_encoding == V4L2_YCBCR_ENC_601;
    const bool quantization_ok =
        quantization == V4L2_QUANTIZATION_DEFAULT ||
        quantization == V4L2_QUANTIZATION_LIM_RANGE;
    return colorspace_ok && encoding_ok && quantization_ok;
}

bool validateCameraBuffer(
    bool error_flag,
    std::size_t index,
    std::size_t buffer_count,
    std::size_t bytesused,
    std::size_t buffer_length,
    std::string& error)
{
    if (index >= buffer_count) {
        error = "camera returned an invalid buffer index";
        return false;
    }
    if (error_flag) {
        error = "camera frame marked with V4L2_BUF_FLAG_ERROR";
        return false;
    }
    if (bytesused == 0) {
        error = "camera returned an empty frame";
        return false;
    }
    if (bytesused > buffer_length) {
        error = "camera returned an invalid frame size";
        return false;
    }
    return true;
}

int dequeueCameraBufferOnce(
    int fd,
    unsigned long request,
    void* buffer,
    const std::function<int(int, unsigned long, void*)>& ioctl_fn)
{
    if (!ioctl_fn) {
        errno = EINVAL;
        return -1;
    }
    return ioctl_fn(fd, request, buffer);
}

CameraReadResult waitForCameraFrame(
    const CameraReadOps& operations,
    int timeout_ms,
    std::string& error)
{
    if (timeout_ms <= 0) {
        error = "camera timeout must be greater than 0 ms";
        return CameraReadResult::Error;
    }
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            error = "camera read timeout after " + std::to_string(timeout_ms) +
                    " ms";
            return CameraReadResult::Timeout;
        }
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   deadline - now)
                                   .count();
        const int remaining_ms =
            static_cast<int>(std::max<long long>(1, remaining));

        int poll_error = 0;
        const int poll_result = operations.poll ? operations.poll(
                                                    remaining_ms, poll_error)
                                                : -1;
        if (poll_result == 0) {
            error = "camera read timeout after " + std::to_string(timeout_ms) +
                    " ms";
            return CameraReadResult::Timeout;
        }
        if (poll_result < 0) {
            if (poll_error == EINTR) {
                continue;
            }
            error = "camera poll failed: " + errnoMessage(poll_error);
            return CameraReadResult::Error;
        }

        int dequeue_error = 0;
        const int dequeue_result = operations.dequeue
                                       ? operations.dequeue(dequeue_error)
                                       : -1;
        if (dequeue_result == 0) {
            return CameraReadResult::FrameReady;
        }
        if (dequeue_error == EAGAIN || dequeue_error == EINTR) {
            continue;
        }
        error = "camera dequeue failed: " + errnoMessage(dequeue_error);
        return CameraReadResult::Error;
    }
}


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
        last_error_kind_ = CameraErrorKind::Fatal;
        return SourceStatus::Error;
    }

    std::vector<std::uint8_t> payload;
    if (!backend_->nextFrame(payload, error)) {
        last_error_kind_ = backend_->lastErrorKind();
        return SourceStatus::Error;
    }
    last_error_kind_ = CameraErrorKind::None;
    info_ = backend_->info();

    cv::Mat image;
    if (!decodePayload(payload, info_, image, error)) {
        last_error_kind_ = CameraErrorKind::BadFrame;
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

CameraErrorKind CameraSource::lastErrorKind() const
{
    return last_error_kind_;
}

ReconnectingCameraSource::ReconnectingCameraSource(
    const CameraOptions& options,
    int max_attempts,
    int reconnect_interval_ms,
    const std::atomic<bool>* external_stop,
    std::string& error)
    : ReconnectingCameraSource(
          options, max_attempts, reconnect_interval_ms, external_stop,
          [](const CameraOptions& camera_options, std::string& open_error) {
              return std::make_unique<CameraSource>(camera_options, open_error);
          },
          error)
{
}

ReconnectingCameraSource::ReconnectingCameraSource(
    const CameraOptions& options,
    int max_attempts,
    int reconnect_interval_ms,
    const std::atomic<bool>* external_stop,
    CameraFactory factory,
    std::string& error)
    : options_(options),
      max_attempts_(max_attempts),
      reconnect_interval_ms_(reconnect_interval_ms),
      external_stop_(external_stop),
      factory_(std::move(factory))
{
    error.clear();
    if (max_attempts_ <= 0) {
        error = "camera reconnect attempts must be greater than 0";
        return;
    }
    if (reconnect_interval_ms_ < 0) {
        error = "camera reconnect interval must not be negative";
        return;
    }
    if (!factory_) {
        error = "camera factory is null";
        return;
    }
    try {
        camera_ = factory_(options_, error);
    } catch (const std::exception& exception) {
        error = std::string("camera open failed: ") + exception.what();
        return;
    }
    if (!camera_) {
        if (error.empty()) {
            error = "camera open failed";
        }
        return;
    }
    info_ = camera_->info();
}

bool ReconnectingCameraSource::shouldStop() const
{
    return external_stop_ != nullptr && external_stop_->load();
}

SourceStatus ReconnectingCameraSource::next(Frame& frame, std::string& error)
{
    error.clear();
    error_kind_ = SourceErrorKind::DataError;
    if (shouldStop()) {
        return SourceStatus::End;
    }

    while (true) {
        if (!camera_) {
            const ReconnectResult reconnect_result = reconnect(error);
            if (reconnect_result == ReconnectResult::Stopped) {
                return SourceStatus::End;
            }
            if (reconnect_result == ReconnectResult::Exhausted) {
                error_kind_ = SourceErrorKind::Fatal;
                return SourceStatus::Error;
            }
            continue;
        }

        if (shouldStop()) {
            return SourceStatus::End;
        }

        Frame camera_frame;
        std::string camera_error;
        const SourceStatus status = camera_->next(camera_frame, camera_error);
        if (status == SourceStatus::Frame) {
            camera_frame.index = next_index_++;
            camera_frame.source = options_.device;
            error_kind_ = SourceErrorKind::DataError;
            frame = std::move(camera_frame);
            return SourceStatus::Frame;
        }
        if (status == SourceStatus::End) {
            return SourceStatus::End;
        }

        const CameraErrorKind camera_kind = camera_->lastErrorKind();
        if (camera_kind == CameraErrorKind::BadFrame) {
            frame.source = options_.device;
            frame.index = next_index_++;
            frame.timestamp = std::chrono::system_clock::now();
            error = camera_error;
            error_kind_ = SourceErrorKind::DataError;
            return SourceStatus::Error;
        }
        if (camera_kind == CameraErrorKind::Fatal) {
            error = camera_error;
            error_kind_ = SourceErrorKind::Fatal;
            return SourceStatus::Error;
        }

        // Timeout/DeviceLost/None: close this instance and reopen with budget.
        camera_.reset();
        if (shouldStop()) {
            return SourceStatus::End;
        }
        const ReconnectResult reconnect_result = reconnect(error);
        if (reconnect_result == ReconnectResult::Stopped) {
            return SourceStatus::End;
        }
        if (reconnect_result == ReconnectResult::Exhausted) {
            error_kind_ = SourceErrorKind::Fatal;
            return SourceStatus::Error;
        }
    }
}

ReconnectingCameraSource::ReconnectResult ReconnectingCameraSource::reconnect(
    std::string& error)
{
    std::string last_error;
    for (int attempt = 0; attempt < max_attempts_; ++attempt) {
        int waited_ms = 0;
        while (waited_ms < reconnect_interval_ms_) {
            if (shouldStop()) {
                error = "camera reconnect stopped";
                return ReconnectResult::Stopped;
            }
            const int chunk_ms =
                std::min(50, reconnect_interval_ms_ - waited_ms);
            std::this_thread::sleep_for(std::chrono::milliseconds(chunk_ms));
            waited_ms += chunk_ms;
        }
        if (shouldStop()) {
            error = "camera reconnect stopped";
            return ReconnectResult::Stopped;
        }

        std::string open_error;
        std::unique_ptr<CameraSource> candidate;
        try {
            candidate = factory_(options_, open_error);
        } catch (const std::exception& exception) {
            open_error = std::string("camera reopen failed: ") +
                         exception.what();
        }
        if (candidate) {
            camera_ = std::move(candidate);
            info_ = camera_->info();
            ++reconnect_count_;
            error.clear();
            return ReconnectResult::Success;
        }
        last_error = open_error.empty() ? "camera reopen failed" : open_error;
    }

    error = "camera reconnect failed after " +
            std::to_string(max_attempts_) + " attempt(s): " + last_error;
    return ReconnectResult::Exhausted;
}

SourceErrorKind ReconnectingCameraSource::errorKind() const
{
    return error_kind_;
}

const CameraInfo& ReconnectingCameraSource::info() const
{
    return info_;
}

std::size_t ReconnectingCameraSource::reconnectCount() const
{
    return reconnect_count_;
}

}  // namespace edge_ai

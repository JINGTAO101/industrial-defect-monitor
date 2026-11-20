#include "ng_image_writer.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

#include <opencv2/imgcodecs.hpp>

namespace edge_ai {
namespace {

std::string errnoText(const std::string& context)
{
    return context + ": " + std::strerror(errno);
}

bool writeAll(int fd, const void* data, std::size_t size, std::string& error)
{
    const auto* bytes = static_cast<const unsigned char*>(data);
    while (size > 0) {
        const ssize_t written = ::write(fd, bytes, size);
        if (written > 0) {
            bytes += written;
            size -= static_cast<std::size_t>(written);
            continue;
        }
        if (written < 0 && errno == EINTR) {
            continue;
        }
        error = errnoText("failed to write NG image data");
        return false;
    }
    return true;
}

class TempFile {
public:
    ~TempFile() { cleanup(); }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    TempFile() = default;

    bool create(const std::string& directory,
                const std::string& stem,
                std::string& error)
    {
        static std::atomic<unsigned long long> counter{0};
        for (int attempt = 0; attempt < 100; ++attempt) {
            const std::string name =
                "." + stem + ".tmp." + std::to_string(::getpid()) + "." +
                std::to_string(counter.fetch_add(1));
            const std::string candidate =
                (std::filesystem::path(directory) / name).string();
            fd_ = ::open(candidate.c_str(),
                         O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
            if (fd_ >= 0) {
                path_ = candidate;
                return true;
            }
            if (errno != EEXIST) {
                error = errnoText("cannot create temporary image file '" +
                                  candidate + "'");
                return false;
            }
            // Name collision: try another suffix.
        }
        error = "cannot create a unique temporary image file in '" +
                directory + "'";
        return false;
    }

    int fd() const { return fd_; }

    bool finish(std::string& error)
    {
        if (fd_ < 0) {
            error = "temporary image file is not open";
            return false;
        }
        if (::fsync(fd_) != 0) {
            error = errnoText("failed to flush temporary image file '" +
                              path_ + "'");
            return false;
        }
        const int close_result = ::close(fd_);
        fd_ = -1;
        if (close_result != 0) {
            error = errnoText("failed to close temporary image file '" +
                              path_ + "'");
            return false;
        }
        return true;
    }

    bool publish(const std::string& directory,
                 const std::string& stem,
                 const std::string& suffix,
                 std::string& saved_path,
                 std::string& error)
    {
        for (int attempt = 0; attempt < 1000; ++attempt) {
            const std::string name =
                attempt == 0 ? stem : stem + "-" + std::to_string(attempt);
            const std::string final_path =
                (std::filesystem::path(directory) / (name + suffix)).string();
            // link(2) publishes without following or replacing an existing
            // directory entry; a dangling symlink at final_path yields EEXIST.
            if (::link(path_.c_str(), final_path.c_str()) == 0) {
                saved_path = final_path;
                if (::unlink(path_.c_str()) == 0) {
                    path_.clear();
                }
                return true;
            }
            if (errno == EEXIST) {
                continue;
            }
            error = errnoText("cannot publish NG image '" + final_path + "'");
            return false;
        }
        error = "cannot find an unused NG image name for '" + stem + "' in '" +
                directory + "'";
        return false;
    }

    void cleanup()
    {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        if (!path_.empty()) {
            ::unlink(path_.c_str());
            path_.clear();
        }
    }

private:
    std::string path_;
    int fd_ = -1;
};

}  // namespace

bool writeNgImage(const cv::Mat& image,
                  const AnnotationInput& annotation,
                  const std::string& source_path,
                  const std::string& event_uid,
                  NgImageMode mode,
                  const std::string& directory,
                  std::string& saved_path,
                  std::string& error)
{
    saved_path.clear();
    error.clear();
    if (directory.empty()) {
        error = "NG image directory is empty";
        return false;
    }
    if (event_uid.empty()) {
        error = "event uid is empty";
        return false;
    }

    std::error_code fs_error;
    std::filesystem::create_directories(directory, fs_error);
    if (fs_error) {
        error = "cannot create NG image directory '" + directory + "': " +
                fs_error.message();
        return false;
    }

    std::string suffix;
    std::vector<unsigned char> annotated_png;
    int source_fd = -1;

    if (mode == NgImageMode::Annotated) {
        suffix = ".png";
        try {
            const cv::Mat annotated = annotateFrame(image, annotation);
            if (annotated.empty()) {
                error = "annotated NG image is empty";
                return false;
            }
            if (!cv::imencode(".png", annotated, annotated_png)) {
                error = "failed to encode annotated NG image as PNG";
                return false;
            }
        } catch (const cv::Exception& exception) {
            error = std::string("failed to encode annotated NG image: ") +
                    exception.what();
            return false;
        }
    } else {
        suffix = std::filesystem::path(source_path).extension().string();
        if (suffix.empty()) {
            suffix = ".img";
        }
        source_fd = ::open(source_path.c_str(), O_RDONLY | O_CLOEXEC);
        if (source_fd < 0) {
            error = errnoText("cannot open input image '" + source_path + "'");
            return false;
        }
    }

    TempFile temp;
    if (!temp.create(directory, event_uid, error)) {
        if (source_fd >= 0) {
            ::close(source_fd);
        }
        return false;
    }

    bool written = false;
    if (mode == NgImageMode::Annotated) {
        written = writeAll(temp.fd(), annotated_png.data(),
                           annotated_png.size(), error);
    } else {
        std::array<unsigned char, 64 * 1024> buffer{};
        while (true) {
            const ssize_t count =
                ::read(source_fd, buffer.data(), buffer.size());
            if (count == 0) {
                written = true;
                break;
            }
            if (count < 0) {
                if (errno == EINTR) {
                    continue;
                }
                error = errnoText("failed to read input image '" +
                                  source_path + "'");
                break;
            }
            if (!writeAll(temp.fd(), buffer.data(),
                          static_cast<std::size_t>(count), error)) {
                break;
            }
        }
        ::close(source_fd);
        source_fd = -1;
    }

    if (!written) {
        return false;  // TempFile destructor removes the partial file.
    }
    if (!temp.finish(error)) {
        return false;
    }
    return temp.publish(directory, event_uid, suffix, saved_path, error);
}

}  // namespace edge_ai

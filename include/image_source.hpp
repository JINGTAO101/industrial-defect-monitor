#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace edge_ai {

// Wall-clock time at which the image was successfully decoded, using
// std::chrono::system_clock. This is a read/acquisition timestamp at the
// software boundary, not EXIF capture time.
using FrameTimestamp = std::chrono::system_clock::time_point;

struct Frame {
    cv::Mat image;       // BGR 8UC3 after successful loading
    std::string source;  // file path
    std::size_t index = 0;
    FrameTimestamp timestamp{};  // set on successful decode
    // Width, height and channel count are available from image.cols,
    // image.rows and image.channels(); they are intentionally not duplicated.
};

enum class SourceStatus { Frame, End, Error };

class ImageSource {
public:
    virtual ~ImageSource() = default;

    // Returns Frame on success, End when exhausted, or Error with a diagnostic
    // in `error`. On Error, `frame.source` identifies the failing file.
    virtual SourceStatus next(Frame& frame, std::string& error) = 0;

    // Number of regular files in a directory source skipped because their
    // extension is not a supported image format.
    virtual std::size_t skippedUnsupported() const { return 0; }
};

bool isSupportedImageExtension(const std::string& path);

// Enumerates supported regular files in `directory`, sorted lexicographically.
// Unsupported extensions are skipped and counted via skipped_unsupported.
// Returns an empty vector and sets error if the directory is invalid or has
// no supported images.
std::vector<std::string> listSupportedImageFiles(
    const std::string& directory,
    std::size_t* skipped_unsupported = nullptr,
    std::string* error = nullptr);

class SingleImageSource final : public ImageSource {
public:
    // Validates existence, regular-file status and extension up front.
    explicit SingleImageSource(std::string path, std::string& error);

    SourceStatus next(Frame& frame, std::string& error) override;

private:
    std::string path_;
    bool delivered_ = false;
    bool failed_ = false;
};

class DirectoryImageSource final : public ImageSource {
public:
    // Validates the directory and requires at least one supported image.
    explicit DirectoryImageSource(std::string directory, std::string& error);

    SourceStatus next(Frame& frame, std::string& error) override;
    std::size_t skippedUnsupported() const override;

private:
    std::vector<std::string> files_;
    std::size_t next_index_ = 0;
    std::size_t skipped_unsupported_ = 0;
};

}  // namespace edge_ai

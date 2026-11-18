#include "image_source.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <system_error>

#include <opencv2/imgcodecs.hpp>

namespace edge_ai {
namespace {

const std::vector<std::string>& supportedExtensions()
{
    static const std::vector<std::string> extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"};
    return extensions;
}

std::string lowerExtension(const std::string& path)
{
    std::string extension = std::filesystem::path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return extension;
}

bool loadFrame(const std::string& path, Frame& frame, std::string& error)
{
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        error = "failed to read image: " + path;
        return false;
    }
    if (image.type() != CV_8UC3) {
        error = "image must be CV_8UC3 BGR: " + path;
        return false;
    }
    frame.image = std::move(image);
    frame.source = path;
    return true;
}

}  // namespace

bool isSupportedImageExtension(const std::string& path)
{
    const std::string extension = lowerExtension(path);
    const auto& extensions = supportedExtensions();
    return std::find(extensions.begin(), extensions.end(), extension) !=
           extensions.end();
}

std::vector<std::string> listSupportedImageFiles(
    const std::string& directory,
    std::size_t* skipped_unsupported,
    std::string* error)
{
    if (skipped_unsupported != nullptr) {
        *skipped_unsupported = 0;
    }
    if (error != nullptr) {
        error->clear();
    }
    std::size_t skipped = 0;
    std::vector<std::string> files;

    std::error_code filesystem_error;
    const std::filesystem::path root(directory);
    if (!std::filesystem::exists(root, filesystem_error) ||
        filesystem_error) {
        if (error != nullptr) {
            *error = "image directory not found: " + directory;
        }
        return files;
    }
    if (!std::filesystem::is_directory(root, filesystem_error) ||
        filesystem_error) {
        if (error != nullptr) {
            *error = "image path is not a directory: " + directory;
        }
        return files;
    }

    for (const auto& entry :
         std::filesystem::directory_iterator(root, filesystem_error)) {
        if (filesystem_error) {
            if (error != nullptr) {
                *error = "failed to enumerate image directory: " + directory +
                         ": " + filesystem_error.message();
            }
            return {};
        }
        std::error_code entry_error;
        if (!entry.is_regular_file(entry_error) || entry_error) {
            continue;
        }
        const std::string path = entry.path().string();
        if (isSupportedImageExtension(path)) {
            files.push_back(path);
        } else {
            ++skipped;
        }
    }

    std::sort(files.begin(), files.end());
    if (skipped_unsupported != nullptr) {
        *skipped_unsupported = skipped;
    }
    if (files.empty() && error != nullptr) {
        *error = "no supported images found in directory: " + directory;
    }
    return files;
}

SingleImageSource::SingleImageSource(std::string path, std::string& error)
    : path_(std::move(path))
{
    error.clear();
    if (path_.empty()) {
        error = "image path is empty";
        return;
    }
    std::error_code filesystem_error;
    if (!std::filesystem::exists(path_, filesystem_error) ||
        filesystem_error) {
        error = "image file not found: " + path_;
        return;
    }
    if (!std::filesystem::is_regular_file(path_, filesystem_error) ||
        filesystem_error) {
        error = "image path is not a regular file: " + path_;
        return;
    }
    if (!isSupportedImageExtension(path_)) {
        error = "unsupported image extension: " + path_;
    }
}

SourceStatus SingleImageSource::next(Frame& frame, std::string& error)
{
    error.clear();
    if (delivered_ || failed_) {
        return SourceStatus::End;
    }
    Frame loaded;
    if (!loadFrame(path_, loaded, error)) {
        failed_ = true;
        frame.source = path_;
        return SourceStatus::Error;
    }
    loaded.index = 0;
    delivered_ = true;
    frame = std::move(loaded);
    return SourceStatus::Frame;
}

DirectoryImageSource::DirectoryImageSource(
    std::string directory,
    std::string& error)
{
    error.clear();
    files_ = listSupportedImageFiles(directory, &skipped_unsupported_, &error);
}

SourceStatus DirectoryImageSource::next(Frame& frame, std::string& error)
{
    error.clear();
    if (next_index_ >= files_.size()) {
        return SourceStatus::End;
    }
    Frame loaded;
    loaded.source = files_[next_index_];
    loaded.index = next_index_;
    ++next_index_;
    if (!loadFrame(loaded.source, loaded, error)) {
        frame.source = loaded.source;
        frame.index = loaded.index;
        return SourceStatus::Error;
    }
    frame = std::move(loaded);
    return SourceStatus::Frame;
}

std::size_t DirectoryImageSource::skippedUnsupported() const
{
    return skipped_unsupported_;
}

}  // namespace edge_ai

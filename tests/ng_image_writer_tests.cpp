#include <atomic>
#include <csignal>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>

#include "ng_image_writer.hpp"

namespace {

using TestFunction = bool (*)(std::string&);

struct TestCase {
    const char* name;
    TestFunction function;
};

bool expect(bool condition, const std::string& message, std::string& error)
{
    if (!condition) {
        error = message;
        return false;
    }
    return true;
}

class TempDir {
public:
    TempDir()
    {
        static std::atomic<int> counter{0};
        path_ = std::filesystem::temp_directory_path() /
                ("edge_ai_ng_" + std::to_string(::getpid()) + "_" +
                 std::to_string(counter.fetch_add(1)));
        std::filesystem::create_directories(path_);
    }

    ~TempDir()
    {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

std::size_t directoryEntryCount(const std::filesystem::path& path)
{
    std::error_code error;
    std::size_t count = 0;
    for (std::filesystem::directory_iterator it(path, error), end;
         it != end && !error; it.increment(error)) {
        ++count;
    }
    return count;
}

void writeSource(const std::filesystem::path& path, std::size_t size)
{
    std::ofstream output(path, std::ios::binary);
    for (std::size_t i = 0; i < size; ++i) {
        output.put(static_cast<char>('A' + (i % 26)));
    }
}

bool testOriginalAndAnnotated(std::string& error)
{
    TempDir temp;
    const std::filesystem::path images = temp.path() / "images";
    const std::filesystem::path source = temp.path() / "source.png";
    writeSource(source, 8192);

    std::string saved;
    std::string write_error;
    if (!expect(edge_ai::writeNgImage(
                    cv::Mat(), edge_ai::AnnotationInput{}, source.string(),
                    "fixed-uid", edge_ai::NgImageMode::Original,
                    images.string(), saved, write_error),
                "original NG image write failed: " + write_error, error)) {
        return false;
    }
    const std::string original_name =
        std::filesystem::path(saved).filename().string();
    if (!expect(original_name.rfind("fixed-uid", 0) == 0,
                "original image is not named after the event id", error)) {
        return false;
    }
    std::ifstream input(source, std::ios::binary);
    std::ifstream saved_input(saved, std::ios::binary);
    const std::vector<char> source_bytes{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    const std::vector<char> saved_bytes{
        std::istreambuf_iterator<char>(saved_input),
        std::istreambuf_iterator<char>()};
    if (!expect(source_bytes == saved_bytes,
                "original mode did not write the source bytes exactly",
                error)) {
        return false;
    }
    if (!expect(directoryEntryCount(images) == 1,
                "original mode left temporary files behind", error)) {
        return false;
    }

    const cv::Mat image(32, 32, CV_8UC3, cv::Scalar(0, 0, 0));
    edge_ai::AnnotationInput annotation;
    annotation.quality_available = true;
    annotation.is_ng = true;
    annotation.detections.push_back({0, "crazing", 0.9f, 1.0f, 2.0f,
                                     20.0f, 25.0f});
    std::string annotated_path;
    if (!expect(edge_ai::writeNgImage(
                    image, annotation, source.string(), "annotated-uid",
                    edge_ai::NgImageMode::Annotated, images.string(),
                    annotated_path, write_error),
                "annotated NG image write failed: " + write_error, error)) {
        return false;
    }
    const std::string annotated_name =
        std::filesystem::path(annotated_path).filename().string();
    if (!expect(annotated_name.rfind("annotated-uid", 0) == 0,
                "annotated image is not named after the event id", error)) {
        return false;
    }
    const cv::Mat decoded = cv::imread(annotated_path, cv::IMREAD_COLOR);
    if (!expect(!decoded.empty() && decoded.cols == 32 && decoded.rows == 32,
                "annotated image is not a readable 32x32 PNG", error)) {
        return false;
    }
    return expect(directoryEntryCount(images) == 2,
                  "annotated mode left temporary files behind", error);
}

bool testDanglingSymlink(std::string& error)
{
    TempDir temp;
    const std::filesystem::path images = temp.path() / "images";
    std::filesystem::create_directories(images);
    const std::filesystem::path link = images / "fixed-uid.png";
    std::error_code link_error;
    std::filesystem::create_symlink("missing-target.png", link, link_error);
    if (!expect(!link_error, "cannot create dangling symlink: " +
                                 link_error.message(),
                error)) {
        return false;
    }

    const cv::Mat image(32, 32, CV_8UC3, cv::Scalar(0, 0, 0));
    edge_ai::AnnotationInput annotation;
    annotation.quality_available = true;
    std::string saved;
    std::string write_error;
    if (!expect(edge_ai::writeNgImage(
                    image, annotation, "", "fixed-uid",
                    edge_ai::NgImageMode::Annotated, images.string(), saved,
                    write_error),
                "annotated write over a dangling symlink failed: " +
                    write_error,
                error)) {
        return false;
    }
    const std::string name =
        std::filesystem::path(saved).filename().string();
    if (!expect(name == "fixed-uid-1.png",
                "dangling symlink was treated as a free name: " + name,
                error)) {
        return false;
    }
    if (!expect(!std::filesystem::exists(images / "missing-target.png"),
                "writer followed the dangling symlink and created its target",
                error)) {
        return false;
    }
    std::error_code status_error;
    return expect(std::filesystem::symlink_status(link, status_error).type() ==
                      std::filesystem::file_type::symlink,
                  "pre-existing dangling symlink was replaced", error);
}

bool testFailureCleansTempFile(std::string& error)
{
    TempDir temp;
    const std::filesystem::path images = temp.path() / "images";
    const std::filesystem::path source = temp.path() / "large.png";
    writeSource(source, 16 * 1024);

    struct rlimit old_limit {};
    if (getrlimit(RLIMIT_FSIZE, &old_limit) != 0) {
        error = "cannot read RLIMIT_FSIZE";
        return false;
    }
    struct sigaction old_action {};
    struct sigaction ignore_action {};
    ignore_action.sa_handler = SIG_IGN;
    sigemptyset(&ignore_action.sa_mask);
    if (sigaction(SIGXFSZ, &ignore_action, &old_action) != 0) {
        error = "cannot ignore SIGXFSZ";
        return false;
    }

    struct rlimit limited = old_limit;
    limited.rlim_cur = 1024;
    if (setrlimit(RLIMIT_FSIZE, &limited) != 0) {
        sigaction(SIGXFSZ, &old_action, nullptr);
        error = "cannot lower RLIMIT_FSIZE";
        return false;
    }

    std::string saved;
    std::string write_error;
    const bool written = edge_ai::writeNgImage(
        cv::Mat(), edge_ai::AnnotationInput{}, source.string(), "fail-uid",
        edge_ai::NgImageMode::Original, images.string(), saved, write_error);

    setrlimit(RLIMIT_FSIZE, &old_limit);
    sigaction(SIGXFSZ, &old_action, nullptr);

    if (!expect(!written, "size-limited image write unexpectedly succeeded",
                error) ||
        !expect(write_error.find("write") != std::string::npos ||
                    write_error.find("flush") != std::string::npos,
                "size-limited write error is unclear: " + write_error,
                error)) {
        return false;
    }
    return expect(directoryEntryCount(images) == 0,
                  "failed image write left a truncated file or temp file",
                  error);
}

}  // namespace

int main(int argc, char* argv[])
{
    const TestCase tests[] = {
        {"ng_image_original_and_annotated", testOriginalAndAnnotated},
        {"ng_image_dangling_symlink", testDanglingSymlink},
        {"ng_image_failure_cleanup", testFailureCleansTempFile},
    };

    const std::string filter = argc > 1 ? argv[1] : "";
    int ran = 0;
    int failures = 0;
    for (const TestCase& test : tests) {
        if (!filter.empty() && filter != test.name) {
            continue;
        }
        ++ran;
        std::string error;
        if (test.function(error)) {
            std::cout << "[PASS] " << test.name << "\n";
        } else {
            ++failures;
            std::cout << "[FAIL] " << test.name << ": " << error << "\n";
        }
    }
    if (ran == 0) {
        std::cerr << "error: unknown NG image test: " << filter << "\n";
        return 2;
    }
    std::cout << ran << " test(s), " << failures << " failure(s)\n";
    return failures == 0 ? 0 : 1;
}

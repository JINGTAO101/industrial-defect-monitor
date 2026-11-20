#pragma once

#include <string>

#include <opencv2/core.hpp>

#include "annotation.hpp"

namespace edge_ai {

enum class NgImageMode { Original, Annotated };

// Writes an NG image into `directory` using `<event_uid>` as the stem:
//   Original: byte-copies `source_path`, preserving its extension.
//   Annotated: encodes `annotateFrame(image, annotation)` as PNG.
//
// The data is first written to an exclusive hidden temp file in the same
// directory. Only after a complete write is it published under a final name;
// publication uses link(2) so an existing file or dangling symlink is never
// followed or overwritten. Names get -1/-2 suffixes on collision. Any failure
// removes the temp file and leaves `saved_path` empty.
bool writeNgImage(const cv::Mat& image,
                  const AnnotationInput& annotation,
                  const std::string& source_path,
                  const std::string& event_uid,
                  NgImageMode mode,
                  const std::string& directory,
                  std::string& saved_path,
                  std::string& error);

}  // namespace edge_ai

#pragma once

#include <string>

namespace edge_ai {

// Returns the lowercase hex SHA256 of a file, or an empty string on error.
std::string sha256File(const std::string& path, std::string& error);

}  // namespace edge_ai

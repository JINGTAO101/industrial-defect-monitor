#include "sha256.hpp"

#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>

#include <openssl/evp.h>

namespace edge_ai {

std::string sha256File(const std::string& path, std::string& error)
{
    error.clear();
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "cannot open file for SHA256: " + path;
        return {};
    }

    using ContextPtr = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
    ContextPtr context(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
    if (!context) {
        error = "EVP_MD_CTX_new failed";
        return {};
    }
    if (EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
        error = "EVP_DigestInit_ex failed";
        return {};
    }

    char buffer[8192];
    while (input) {
        input.read(buffer, sizeof(buffer));
        const std::streamsize count = input.gcount();
        if (count > 0 &&
            EVP_DigestUpdate(context.get(), buffer,
                             static_cast<std::size_t>(count)) != 1) {
            error = "EVP_DigestUpdate failed";
            return {};
        }
    }
    if (input.bad()) {
        error = "I/O error while reading file for SHA256: " + path;
        return {};
    }

    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_length = 0;
    if (EVP_DigestFinal_ex(context.get(), digest, &digest_length) != 1) {
        error = "EVP_DigestFinal_ex failed";
        return {};
    }

    std::ostringstream hex;
    hex << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < digest_length; ++i) {
        hex << std::setw(2) << static_cast<int>(digest[i]);
    }
    return hex.str();
}

}  // namespace edge_ai

#include "model_contract.hpp"

#include <array>
#include <cctype>
#include <sstream>

namespace edge_ai {
namespace {

std::string expectedInputShape()
{
    return "[1, 3, 640, 640]";
}

std::string expectedOutputShape()
{
    return "[1, 10, 8400]";
}

bool isSpace(char c)
{
    return std::isspace(static_cast<unsigned char>(c)) != 0;
}

}  // namespace

std::string formatShape(const std::vector<int64_t>& shape)
{
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        os << shape[i];
    }
    os << "]";
    return os.str();
}

std::string validateModelContract(const ModelContract& contract)
{
    if (contract.input_count != 1) {
        return "input count mismatch: expected 1, got " +
               std::to_string(contract.input_count);
    }
    if (contract.output_count != 1) {
        return "output count mismatch: expected 1, got " +
               std::to_string(contract.output_count);
    }
    if (contract.input.name != "images") {
        return "input name mismatch: expected 'images', got '" +
               contract.input.name + "'";
    }
    if (contract.input.shape != std::vector<int64_t>{1, 3, kInputSize, kInputSize}) {
        return "input shape mismatch: expected " + expectedInputShape() +
               ", got " + formatShape(contract.input.shape);
    }
    if (contract.input.type != "tensor(float)") {
        return "input type mismatch: expected 'tensor(float)', got '" +
               contract.input.type + "'";
    }
    if (contract.output.name != "output0") {
        return "output name mismatch: expected 'output0', got '" +
               contract.output.name + "'";
    }
    if (contract.output.shape !=
        std::vector<int64_t>{1, kOutputChannels, kNumPredictions}) {
        return "output shape mismatch: expected " + expectedOutputShape() +
               ", got " + formatShape(contract.output.shape);
    }
    if (contract.output.type != "tensor(float)") {
        return "output type mismatch: expected 'tensor(float)', got '" +
               contract.output.type + "'";
    }
    if (contract.class_names.size() != static_cast<std::size_t>(kNumClasses)) {
        return "class count mismatch: expected " + std::to_string(kNumClasses) +
               ", got " + std::to_string(contract.class_names.size());
    }

    const std::array<std::string, kNumClasses> expected_names = {
        "crazing", "inclusion", "patches",
        "pitted_surface", "rolled-in_scale", "scratches"};
    for (int id = 0; id < kNumClasses; ++id) {
        const auto it = contract.class_names.find(id);
        if (it == contract.class_names.end()) {
            return "class id " + std::to_string(id) + " is missing from metadata";
        }
        if (it->second != expected_names[static_cast<std::size_t>(id)]) {
            return "class id " + std::to_string(id) +
                   " name mismatch: expected '" +
                   expected_names[static_cast<std::size_t>(id)] + "', got '" +
                   it->second + "'";
        }
    }
    return {};
}

bool parseClassNamesMetadata(
    const std::string& raw,
    std::map<int, std::string>& class_names,
    std::string& error)
{
    class_names.clear();
    std::size_t pos = 0;
    const auto skip_ws = [&]() {
        while (pos < raw.size() && isSpace(raw[pos])) {
            ++pos;
        }
    };
    const auto fail = [&](const std::string& message) {
        error = message + " at offset " + std::to_string(pos);
        return false;
    };

    skip_ws();
    if (pos >= raw.size() || raw[pos] != '{') {
        return fail("class metadata is not a dict");
    }
    ++pos;
    skip_ws();
    if (pos < raw.size() && raw[pos] == '}') {
        ++pos;
        skip_ws();
        if (pos != raw.size()) {
            return fail("trailing characters after class metadata");
        }
        return true;
    }

    while (true) {
        skip_ws();
        const std::size_t id_start = pos;
        while (pos < raw.size() &&
               std::isdigit(static_cast<unsigned char>(raw[pos])) != 0) {
            ++pos;
        }
        if (id_start == pos) {
            return fail("expected a non-negative class id");
        }
        int id = 0;
        try {
            id = std::stoi(raw.substr(id_start, pos - id_start));
        } catch (const std::exception&) {
            return fail("class id is out of range");
        }

        skip_ws();
        if (pos >= raw.size() || raw[pos] != ':') {
            return fail("expected ':' after class id");
        }
        ++pos;
        skip_ws();
        if (pos >= raw.size() || (raw[pos] != '\'' && raw[pos] != '"')) {
            return fail("expected a quoted class name");
        }
        const char quote = raw[pos++];
        std::string name;
        bool closed = false;
        while (pos < raw.size()) {
            const char c = raw[pos++];
            if (c == '\\' && pos < raw.size()) {
                name.push_back(raw[pos++]);
                continue;
            }
            if (c == quote) {
                closed = true;
                break;
            }
            name.push_back(c);
        }
        if (!closed) {
            return fail("unterminated class name");
        }
        if (!class_names.emplace(id, name).second) {
            return fail("duplicate class id " + std::to_string(id));
        }

        skip_ws();
        if (pos < raw.size() && raw[pos] == ',') {
            ++pos;
            continue;
        }
        if (pos < raw.size() && raw[pos] == '}') {
            ++pos;
            break;
        }
        return fail("expected ',' or '}' after class entry");
    }

    skip_ws();
    if (pos != raw.size()) {
        return fail("trailing characters after class metadata");
    }
    int expected = 0;
    for (const auto& entry : class_names) {
        if (entry.first != expected) {
            error = "class ids are not contiguous from 0: expected " +
                    std::to_string(expected) + ", got " +
                    std::to_string(entry.first);
            return false;
        }
        ++expected;
    }
    return true;
}

}  // namespace edge_ai

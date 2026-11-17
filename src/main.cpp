#include <iostream>
#include <string>
#include <string_view>

#include "system_info.hpp"

#ifndef EDGE_AI_VERSION
#define EDGE_AI_VERSION "0.1.0"
#endif

namespace {

constexpr std::string_view kProgram = "edge_ai";

void printUsage(std::ostream& os)
{
    os << "edge_ai - Industrial defect monitoring system (edge AI)\n"
       << "\n"
       << "Usage:\n"
       << "  edge_ai [OPTION]\n"
       << "\n"
       << "Options:\n"
       << "  -h, --help     Show this help message and exit\n"
       << "  -V, --version  Show version information and exit\n"
       << "\n"
       << "With no options, edge_ai prints its startup banner.\n";
}

int printError(const std::string& message)
{
    std::cerr << kProgram << ": " << message << "\n"
              << "Try '" << kProgram << " --help' for more information.\n";
    return 2;
}

}  // namespace

int main(int argc, char* argv[])
{
    bool show_help = false;
    bool show_version = false;

    // Validate every argument before acting on --help/--version, so that an
    // invalid argument is never silently ignored (e.g. "edge_ai --help --bogus").
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            show_help = true;
        } else if (arg == "-V" || arg == "--version") {
            show_version = true;
        } else if (!arg.empty() && arg.front() == '-') {
            return printError("unknown option '" + std::string(arg) + "'");
        } else {
            return printError("unexpected argument '" + std::string(arg) + "'");
        }
    }

    if (show_help) {
        printUsage(std::cout);
        return 0;
    }

    if (show_version) {
        std::cout << kProgram << " " << EDGE_AI_VERSION << "\n";
        return 0;
    }

    std::cout << "====================================\n";
    std::cout << " Industrial Defect Monitor System\n";
    std::cout << "====================================\n";
    std::cout << getSystemMessage() << "\n";

    return 0;
}

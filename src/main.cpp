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
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            printUsage(std::cout);
            return 0;
        }

        if (arg == "-V" || arg == "--version") {
            std::cout << kProgram << " " << EDGE_AI_VERSION << "\n";
            return 0;
        }

        if (!arg.empty() && arg.front() == '-') {
            return printError("unknown option '" + std::string(arg) + "'");
        }

        return printError("unexpected argument '" + std::string(arg) + "'");
    }

    std::cout << "====================================\n";
    std::cout << " Industrial Defect Monitor System\n";
    std::cout << "====================================\n";
    std::cout << getSystemMessage() << "\n";

    return 0;
}

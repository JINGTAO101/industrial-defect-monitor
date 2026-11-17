cmake_minimum_required(VERSION 3.20)

project(edge_ai_system
    VERSION 0.1.0
    DESCRIPTION "Industrial defect monitoring system for edge devices"
    LANGUAGES CXX
)

# C++17，禁止编译器扩展，导出 compile_commands.json 供 IDE/工具使用
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 可执行文件统一输出到 build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 基础信息模块（后续可替换/扩展为真正的系统状态与环境探测）
add_library(system_info STATIC
    src/system_info.cpp
)

target_include_directories(system_info
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# 主程序 edge_ai
add_executable(edge_ai
    src/main.cpp
)

target_link_libraries(edge_ai
    PRIVATE
        system_info
)

target_compile_features(edge_ai PRIVATE cxx_std_17)

target_compile_definitions(edge_ai
    PRIVATE
        EDGE_AI_VERSION="${PROJECT_VERSION}"
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(system_info PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(edge_ai PRIVATE -Wall -Wextra -Wpedantic)
endif()

# OpenCV 本阶段不参与编译，仅探测并记录版本，
# 供后续摄像头/图像预处理/推理模块使用。
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION} (optional at this stage)")
else()
    message(STATUS "OpenCV not found (optional at this stage)")
endif()

include(CTest)
if(BUILD_TESTING)
    add_test(
        NAME cli_behavior
        COMMAND ${CMAKE_COMMAND}
            -DBIN=$<TARGET_FILE:edge_ai>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_tests.cmake
    )
endif()

install(TARGETS edge_ai RUNTIME DESTINATION bin)

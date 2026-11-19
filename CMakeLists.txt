cmake_minimum_required(VERSION 3.20)

project(edge_ai_system
    VERSION 0.2.0
    DESCRIPTION "Industrial defect monitoring system for edge devices"
    LANGUAGES CXX
)

# C++17，禁止编译器扩展，导出 compile_commands.json 供 IDE/工具使用
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# We add exactly one RUNPATH below; CMake's automatic RPATH generation can
# produce empty entries (which make the loader search the current directory).
set(CMAKE_SKIP_RPATH TRUE)

# 可执行文件统一输出到 build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---- 依赖 ----------------------------------------------------------------

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui dnn)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

# ONNX Runtime C++ CPU 包：解压后包含 include/ 与 lib/。
# 用法：cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.32.0 ...
set(ONNXRUNTIME_ROOT "" CACHE PATH "Root of the ONNX Runtime C++ package")
if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS
        "${ONNXRUNTIME_ROOT}/include"
        "${ONNXRUNTIME_ROOT}/include/onnxruntime"
        "$ENV{ONNXRUNTIME_ROOT}/include"
        "$ENV{ONNXRUNTIME_ROOT}/include/onnxruntime"
)
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS
        "${ONNXRUNTIME_ROOT}/lib"
        "${ONNXRUNTIME_ROOT}/lib64"
        "$ENV{ONNXRUNTIME_ROOT}/lib"
        "$ENV{ONNXRUNTIME_ROOT}/lib64"
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "ONNX Runtime C++ was not found. Download the CPU package "
        "(onnxruntime-linux-x64-<version>.tgz) and pass "
        "-DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-<version>")
endif()
get_filename_component(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
message(STATUS "ONNX Runtime headers: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")

# ---- 公共编译选项 ---------------------------------------------------------

function(edge_ai_enable_warnings target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

# ---- 基础模块 -------------------------------------------------------------

add_library(system_info STATIC
    src/system_info.cpp
)
target_include_directories(system_info PUBLIC ${PROJECT_SOURCE_DIR}/include)
edge_ai_enable_warnings(system_info)

# 预处理/后处理/契约校验：不依赖 ONNX Runtime，可单独测试。
add_library(edge_ai_core STATIC
    src/model_contract.cpp
    src/image_preprocess.cpp
    src/postprocess.cpp
    src/quality.cpp
    src/image_source.cpp
    src/camera_source.cpp
    src/annotation.cpp
    src/preview_window.cpp
)
target_include_directories(edge_ai_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(edge_ai_core PUBLIC ${OpenCV_LIBS})
target_compile_features(edge_ai_core PUBLIC cxx_std_17)
edge_ai_enable_warnings(edge_ai_core)

# ONNX Runtime C++ 推理封装。
add_library(edge_ai_inference STATIC
    src/model_runner.cpp
    src/sha256.cpp
    src/batch_processor.cpp
    src/threaded_pipeline.cpp
)
target_include_directories(edge_ai_inference PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(edge_ai_inference
    PUBLIC
        edge_ai_core
        ${ONNXRUNTIME_LIBRARY}
        OpenSSL::Crypto
)
edge_ai_enable_warnings(edge_ai_inference)

add_executable(edge_ai
    src/main.cpp
)
target_link_libraries(edge_ai PRIVATE edge_ai_inference system_info)
target_compile_definitions(edge_ai PRIVATE EDGE_AI_VERSION="${PROJECT_VERSION}")
edge_ai_enable_warnings(edge_ai)

# 让 build/bin/edge_ai 直接找到 ONNX Runtime 共享库。
# CMAKE_SKIP_RPATH 已关闭 CMake 的自动 RPATH，这里只加入一个明确目录。
target_link_options(edge_ai PRIVATE "-Wl,-rpath,${ONNXRUNTIME_LIB_DIR}")

# ---- 可选的真实模型集成测试配置 ------------------------------------------

set(EDGE_AI_MODEL_PATH "" CACHE FILEPATH
    "Path to the verified neu_det_yolov8n_best.onnx (optional, enables integration tests)")
set(EDGE_AI_TEST_IMAGES_DIR "" CACHE PATH
    "Directory containing the six NEU-DET test JPGs (optional, enables reference comparison)")
set(EDGE_AI_PYTHON "" CACHE FILEPATH
    "Python interpreter with tools/requirements-reference.txt installed")

# ---- 测试 -----------------------------------------------------------------

include(CTest)
if(BUILD_TESTING)
    add_executable(edge_ai_tests
        tests/unit_tests.cpp
    )
    target_link_libraries(edge_ai_tests PRIVATE edge_ai_core)
    target_include_directories(edge_ai_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_compile_definitions(edge_ai_tests PRIVATE
        EDGE_AI_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
    edge_ai_enable_warnings(edge_ai_tests)

    add_executable(edge_ai_batch_tests
        tests/batch_processor_tests.cpp
    )
    target_link_libraries(edge_ai_batch_tests PRIVATE edge_ai_inference)
    target_include_directories(edge_ai_batch_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_compile_definitions(edge_ai_batch_tests PRIVATE
        EDGE_AI_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
    target_link_options(edge_ai_batch_tests PRIVATE
        "-Wl,-rpath,${ONNXRUNTIME_LIB_DIR}")
    edge_ai_enable_warnings(edge_ai_batch_tests)
    add_test(NAME batch_source_exception COMMAND edge_ai_batch_tests)
    # A regression to the old retry-forever behavior must fail promptly.
    set_tests_properties(batch_source_exception PROPERTIES TIMEOUT 10)

    add_executable(edge_ai_threaded_tests
        tests/threaded_pipeline_tests.cpp
    )
    target_link_libraries(edge_ai_threaded_tests PRIVATE edge_ai_inference)
    target_include_directories(edge_ai_threaded_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_compile_definitions(edge_ai_threaded_tests PRIVATE
        EDGE_AI_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
    target_link_options(edge_ai_threaded_tests PRIVATE
        "-Wl,-rpath,${ONNXRUNTIME_LIB_DIR}")
    edge_ai_enable_warnings(edge_ai_threaded_tests)
    foreach(threaded_test
            threaded_normal_drain
            threaded_full_backpressure
            threaded_preview_image_included
            threaded_external_stop_drains
            threaded_fatal_wakes
            threaded_start_failure_inference
            threaded_start_failure_result
            threaded_wait_gate_stale_flag)
        add_test(NAME ${threaded_test} COMMAND edge_ai_threaded_tests ${threaded_test})
        set_tests_properties(${threaded_test} PROPERTIES TIMEOUT 10)
    endforeach()

    add_executable(edge_ai_queue_tests
        tests/queue_tests.cpp
    )
    target_include_directories(edge_ai_queue_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(edge_ai_queue_tests PRIVATE Threads::Threads)
    edge_ai_enable_warnings(edge_ai_queue_tests)

    foreach(queue_test
            queue_reject_zero_capacity
            queue_full_blocks
            queue_empty_blocks
            queue_close_drains
            queue_close_wakes_waiters
            queue_concurrent_no_loss)
        add_test(NAME ${queue_test} COMMAND edge_ai_queue_tests ${queue_test})
        set_tests_properties(${queue_test} PROPERTIES TIMEOUT 10)
    endforeach()

    foreach(test_name
            preprocess_non_square
            postprocess_empty
            postprocess_nms_same_class
            postprocess_nms_cross_class
            postprocess_coordinate_restore
            postprocess_nonfinite
            postprocess_score_range
            postprocess_shape_mismatch
            class_mapping_unknown
            contract_validation
            class_metadata_parser
            quality_no_defects
            quality_single_defect
            quality_below_threshold
            quality_multiple_defects
            quality_boundary
            quality_invalid_threshold
            image_source_single
            image_source_directory
            camera_missing_device
            camera_timeout
            camera_invalid_options
            camera_frame_conversion
            camera_buffer_validation
            camera_busy_error
            camera_read_deadline
            camera_read_stop
            camera_source_external_stop
            camera_colorimetry
            camera_reconnect_success
            camera_reconnect_exhausted
            camera_reconnect_open_error
            camera_reconnect_no_frame_budget
            camera_reconnect_stop
            camera_reconnect_stop_during_open
            camera_bad_frame_sequence
            annotation_ng_boxes
            annotation_error_frame
            annotation_label_avoids_banner
            annotation_label_right_edge
            annotation_save_png
            preview_drops_frames
            preview_close_stops
            preview_closed_without_frames
            preview_error_stops)
        add_test(NAME unit_${test_name} COMMAND edge_ai_tests ${test_name})
    endforeach()

    add_test(
        NAME cli_behavior
        COMMAND ${CMAKE_COMMAND}
            -DBIN=$<TARGET_FILE:edge_ai>
            -DFIXTURES=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures
            -DTEST_DIR=${CMAKE_BINARY_DIR}/cli-tests
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_tests.cmake
    )

    add_test(
        NAME cli_inference_failures
        COMMAND ${CMAKE_COMMAND}
            -DBIN=$<TARGET_FILE:edge_ai>
            -DFIXTURES=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures
            -DREAL_MODEL=${EDGE_AI_MODEL_PATH}
            -DTEST_DIR=${CMAKE_BINARY_DIR}/cli-inference-failure-tests
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_inference_tests.cmake
    )

    add_test(
        NAME cli_empty_detections
        COMMAND ${CMAKE_COMMAND}
            -DBIN=$<TARGET_FILE:edge_ai>
            -DFIXTURES=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_empty_detections.cmake
    )

    add_test(
        NAME cli_path_conflicts
        COMMAND ${CMAKE_COMMAND}
            -DBIN=$<TARGET_FILE:edge_ai>
            -DFIXTURES=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures
            -DTEST_DIR=${CMAKE_BINARY_DIR}/path-conflict-tests
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_path_conflict_tests.cmake
    )

    add_test(
        NAME cli_batch
        COMMAND ${CMAKE_COMMAND}
            -DBIN=$<TARGET_FILE:edge_ai>
            -DFIXTURES=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures
            -DTEST_DIR=${CMAKE_BINARY_DIR}/cli-batch-tests
            -DREAL_MODEL=${EDGE_AI_MODEL_PATH}
            -DREAL_IMAGE_DIR=${EDGE_AI_TEST_IMAGES_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_batch_tests.cmake
    )

    add_test(
        NAME cli_camera
        COMMAND ${CMAKE_COMMAND}
            -DBIN=$<TARGET_FILE:edge_ai>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_camera_tests.cmake
    )
    set_tests_properties(cli_camera PROPERTIES TIMEOUT 10)

    add_test(
        NAME cli_signal
        COMMAND ${CMAKE_COMMAND}
            -DBIN=$<TARGET_FILE:edge_ai>
            -DFIXTURES=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures
            -DTEST_DIR=${CMAKE_BINARY_DIR}/cli-signal-tests
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_signal_tests.cmake
    )
    set_tests_properties(cli_signal PROPERTIES TIMEOUT 60)

    add_test(
        NAME cli_annotation
        COMMAND ${CMAKE_COMMAND}
            -DBIN=$<TARGET_FILE:edge_ai>
            -DFIXTURES=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures
            -DTEST_DIR=${CMAKE_BINARY_DIR}/cli-annotation-tests
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_annotation_tests.cmake
    )
    set_tests_properties(cli_annotation PROPERTIES TIMEOUT 30)

    if(EDGE_AI_MODEL_PATH AND EDGE_AI_TEST_IMAGES_DIR AND EDGE_AI_PYTHON)
        add_test(
            NAME cpp_reference_comparison
            COMMAND ${EDGE_AI_PYTHON} -I
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/compare_cpp_reference.py
                --edge-ai $<TARGET_FILE:edge_ai>
                --model ${EDGE_AI_MODEL_PATH}
                --image-dir ${EDGE_AI_TEST_IMAGES_DIR}
                --reference-results ${CMAKE_CURRENT_SOURCE_DIR}/docs/reference-inference-results.json
                --out-dir ${CMAKE_BINARY_DIR}/reference-comparison
        )

        add_test(
            NAME cpp_reference_rejects_incomplete_inputs
            COMMAND ${CMAKE_COMMAND}
                -DPYTHON=${EDGE_AI_PYTHON}
                -DEDGE_AI=$<TARGET_FILE:edge_ai>
                -DMODEL=${EDGE_AI_MODEL_PATH}
                -DIMAGE_DIR=${EDGE_AI_TEST_IMAGES_DIR}
                -DREFERENCE_RESULTS=${CMAKE_CURRENT_SOURCE_DIR}/docs/reference-inference-results.json
                -DSCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/tools/compare_cpp_reference.py
                -DTEST_DIR=${CMAKE_BINARY_DIR}/reference-guard-tests
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_reference_guard_tests.cmake
        )

        add_test(
            NAME threaded_batch_reference
            COMMAND ${CMAKE_COMMAND}
                -DBIN=$<TARGET_FILE:edge_ai>
                -DPYTHON=${EDGE_AI_PYTHON}
                -DMODEL=${EDGE_AI_MODEL_PATH}
                -DIMAGE_DIR=${EDGE_AI_TEST_IMAGES_DIR}
                -DREFERENCE_RESULTS=${CMAKE_CURRENT_SOURCE_DIR}/docs/reference-inference-results.json
                -DCOMPARE_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/tools/compare_batch_reference.py
                -DTEST_DIR=${CMAKE_BINARY_DIR}/threaded-batch-reference
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/threaded_batch_reference.cmake
        )
        set_tests_properties(threaded_batch_reference PROPERTIES TIMEOUT 30)
    endif()
endif()

install(TARGETS edge_ai RUNTIME DESTINATION bin)

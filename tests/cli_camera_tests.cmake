# Camera CLI validation and missing-device tests. No physical camera required.
#
# Invoked by CTest:
#   cmake -DBIN=<edge_ai> -P cli_camera_tests.cmake

if(NOT DEFINED BIN OR NOT EXISTS "${BIN}")
    message(FATAL_ERROR "edge_ai executable not found: ${BIN}")
endif()

function(expect_camera_error label expected_rc expected_regex)
    execute_process(
        COMMAND "${BIN}" ${ARGN}
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
    )
    if(NOT rc EQUAL expected_rc)
        message(FATAL_ERROR
            "${label}: expected exit ${expected_rc}, got ${rc}; stderr: ${err}")
    endif()
    if(NOT err MATCHES "${expected_regex}")
        message(FATAL_ERROR
            "${label}: stderr does not match '${expected_regex}'; got: ${err}")
    endif()
    if(NOT out STREQUAL "")
        message(FATAL_ERROR "${label}: expected empty stdout, got: ${out}")
    endif()
endfunction()

expect_camera_error(
    "missing camera device"
    1
    "camera device not found"
    --camera "/dev/edge_ai_missing_camera"
    --capture-only
    --max-frames 1
)

expect_camera_error(
    "capture-only without camera"
    2
    "requires --camera"
    --capture-only
    --max-frames 1
)

expect_camera_error(
    "camera without model or capture-only"
    2
    "requires --model or --capture-only"
    --camera "/dev/video0"
    --max-frames 1
)

expect_camera_error(
    "missing camera for inference"
    1
    "camera device not found"
    --camera "/dev/edge_ai_missing_camera"
    --model "/tmp/does-not-matter.onnx"
    --max-frames 1
)

expect_camera_error(
    "reconnect options with capture-only"
    2
    "require camera inference"
    --camera "/dev/video0"
    --capture-only
    --camera-reconnect-attempts 2
)

expect_camera_error(
    "camera inference with image"
    2
    "cannot be combined with --image"
    --camera "/dev/video0"
    --model "/tmp/model.onnx"
    --image "/tmp/image.jpg"
)

expect_camera_error(
    "camera inference queue capacity zero"
    2
    "greater than 0"
    --camera "/dev/video0"
    --model "/tmp/model.onnx"
    --queue-capacity 0
)

expect_camera_error(
    "max-frames zero"
    2
    "greater than 0"
    --camera "/dev/video0"
    --capture-only
    --max-frames 0
)

expect_camera_error(
    "unsupported camera format"
    2
    "unsupported camera format"
    --camera "/dev/video0"
    --capture-only
    --camera-format RGB24
)

expect_camera_error(
    "camera width zero"
    2
    "greater than 0"
    --camera "/dev/video0"
    --capture-only
    --camera-width 0
)

expect_camera_error(
    "capture-only with model"
    2
    "cannot be combined with model/image"
    --camera "/dev/video0"
    --capture-only
    --model "/tmp/model.onnx"
)

expect_camera_error(
    "capture-only with json-out"
    2
    "does not support --json-out"
    --camera "/dev/video0"
    --capture-only
    --json-out "/tmp/camera.json"
)

message(STATUS "All camera CLI checks passed")

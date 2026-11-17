# End-to-end empty-detection test with a valid-contract synthetic ONNX model.
# Invoked by CTest as:
#   cmake -DBIN=<edge_ai> -DFIXTURES=<tests/fixtures> -P cli_empty_detections.cmake

if(NOT DEFINED BIN OR NOT EXISTS "${BIN}")
    message(FATAL_ERROR "edge_ai executable not found: ${BIN}")
endif()
if(NOT DEFINED FIXTURES OR NOT EXISTS "${FIXTURES}/zero_output.onnx")
    message(FATAL_ERROR "FIXTURES is not set or zero_output.onnx is missing: ${FIXTURES}")
endif()

execute_process(
    COMMAND "${BIN}"
        --model "${FIXTURES}/zero_output.onnx"
        --image "${FIXTURES}/non_square_320x200.png"
        --conf 0.25 --iou 0.45
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "empty-detection run exited with ${rc}; stderr: ${err}")
endif()
if(NOT out MATCHES "detections: 0")
    message(FATAL_ERROR "expected 'detections: 0', got: ${out}")
endif()
if(NOT err STREQUAL "")
    message(FATAL_ERROR "empty-detection run wrote unexpected stderr: ${err}")
endif()
message(STATUS "Empty-detection CLI check passed")

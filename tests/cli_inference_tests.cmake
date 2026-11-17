# Failure-path tests for the C++ inference CLI.
# Invoked by CTest as:
#   cmake -DBIN=<edge_ai> -DFIXTURES=<tests/fixtures> [-DREAL_MODEL=<model.onnx>] -P cli_inference_tests.cmake

if(NOT DEFINED BIN)
    message(FATAL_ERROR "BIN is not defined")
endif()
if(NOT EXISTS "${BIN}")
    message(FATAL_ERROR "edge_ai executable not found: ${BIN}")
endif()
if(NOT DEFINED FIXTURES OR NOT EXISTS "${FIXTURES}/non_square_320x200.png")
    message(FATAL_ERROR "FIXTURES is not set or missing fixtures: ${FIXTURES}")
endif()

function(expect_failure label expected_regex)
    execute_process(
        COMMAND "${BIN}" ${ARGN}
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
    )
    if(rc EQUAL 0)
        message(FATAL_ERROR "${label}: expected non-zero exit, got 0 (stdout: ${out})")
    endif()
    if(NOT err MATCHES "${expected_regex}")
        message(FATAL_ERROR "${label}: stderr does not match '${expected_regex}'; got: ${err}")
    endif()
    message(STATUS "${label}: rc=${rc}; stderr: ${err}")
endfunction()

expect_failure(
    "missing model"
    "model file not found"
    --model "${FIXTURES}/does_not_exist.onnx"
    --image "${FIXTURES}/non_square_320x200.png"
)

expect_failure(
    "corrupt model"
    "failed to load ONNX model"
    --model "${FIXTURES}/corrupt_model.bin"
    --image "${FIXTURES}/non_square_320x200.png"
)

expect_failure(
    "wrong input contract"
    "input shape mismatch"
    --model "${FIXTURES}/wrong_input.onnx"
    --image "${FIXTURES}/non_square_320x200.png"
)

expect_failure(
    "wrong output contract"
    "output shape mismatch"
    --model "${FIXTURES}/wrong_output.onnx"
    --image "${FIXTURES}/non_square_320x200.png"
)

if(DEFINED REAL_MODEL AND NOT REAL_MODEL STREQUAL "")
    expect_failure(
        "invalid image"
        "failed to read image"
        --model "${REAL_MODEL}"
        --image "${FIXTURES}/corrupt_model.bin"
    )
else()
    message(STATUS "invalid image test: SKIPPED (REAL_MODEL not provided)")
endif()

message(STATUS "All CLI inference failure checks passed")

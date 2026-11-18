# CLI behaviour tests for edge_ai, executed by CTest via `cmake -P`.
# Checks real process exit codes and stdout/stderr contents.

if(NOT DEFINED BIN)
    message(FATAL_ERROR "BIN is not defined (expected path to the edge_ai executable)")
endif()

if(NOT EXISTS "${BIN}")
    message(FATAL_ERROR "edge_ai executable not found: ${BIN}")
endif()
if(NOT DEFINED FIXTURES OR NOT EXISTS "${FIXTURES}/non_square_320x200.png")
    message(FATAL_ERROR "FIXTURES is not set or missing non_square_320x200.png")
endif()

# 1. --help must exit 0 and print usage.
execute_process(
    COMMAND "${BIN}" --help
    RESULT_VARIABLE help_rc
    OUTPUT_VARIABLE help_out
    ERROR_VARIABLE help_err
)
if(NOT help_rc EQUAL 0)
    message(FATAL_ERROR "--help exited with ${help_rc}; stderr: ${help_err}")
endif()
if(NOT help_out MATCHES "Usage:")
    message(FATAL_ERROR "--help output does not contain 'Usage:'; got: ${help_out}")
endif()
if(NOT help_out MATCHES "--help")
    message(FATAL_ERROR "--help output does not document '--help'; got: ${help_out}")
endif()

# 2. Unknown option must exit non-zero and explain the error on stderr.
execute_process(
    COMMAND "${BIN}" --definitely-not-an-option
    RESULT_VARIABLE bad_rc
    OUTPUT_VARIABLE bad_out
    ERROR_VARIABLE bad_err
)
if(bad_rc EQUAL 0)
    message(FATAL_ERROR "unknown option unexpectedly exited with 0")
endif()
if(NOT bad_err MATCHES "unknown option")
    message(FATAL_ERROR "unknown option stderr does not contain 'unknown option'; got: ${bad_err}")
endif()
if(NOT bad_err MATCHES "--help")
    message(FATAL_ERROR "unknown option stderr does not point at '--help'; got: ${bad_err}")
endif()

# 3. An invalid option must take precedence over --help/--version, regardless
#    of the order in which the arguments appear.
function(expect_invalid_overrides_flag)
    execute_process(
        COMMAND "${BIN}" ${ARGN}
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
    )
    if(rc EQUAL 0)
        message(FATAL_ERROR "expected non-zero exit for arguments '${ARGN}', got 0")
    endif()
    if(NOT err MATCHES "unknown option")
        message(FATAL_ERROR "arguments '${ARGN}': stderr missing 'unknown option'; got: ${err}")
    endif()
    if(NOT out STREQUAL "")
        message(FATAL_ERROR "arguments '${ARGN}': expected empty stdout, got: ${out}")
    endif()
endfunction()

expect_invalid_overrides_flag(--help --bogus)
expect_invalid_overrides_flag(--version --bogus)
expect_invalid_overrides_flag(--bogus --help)
expect_invalid_overrides_flag(--bogus --version)

# 3b. A value-taking option must not consume a following option as its value.
function(expect_missing_value)
    execute_process(
        COMMAND "${BIN}" ${ARGN}
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
    )
    if(rc EQUAL 0)
        message(FATAL_ERROR "arguments '${ARGN}': expected non-zero exit, got 0")
    endif()
    if(NOT err MATCHES "missing value")
        message(FATAL_ERROR "arguments '${ARGN}': stderr missing 'missing value'; got: ${err}")
    endif()
    if(NOT out STREQUAL "")
        message(FATAL_ERROR "arguments '${ARGN}': expected empty stdout, got: ${out}")
    endif()
endfunction()

expect_missing_value(--model)
expect_missing_value(--image)
expect_missing_value(--json-out)
expect_missing_value(--dump-input)
expect_missing_value(--conf)
expect_missing_value(--iou)
expect_missing_value(--help --model --bogus)
expect_missing_value(--image --help)
expect_missing_value(--version --conf --bogus)

# 3c. A value that starts with '-' is passed with the --option=value form.
execute_process(
    COMMAND "${BIN}"
        --model=--bogus
        --image=${FIXTURES}/non_square_320x200.png
    RESULT_VARIABLE equals_rc
    OUTPUT_VARIABLE equals_out
    ERROR_VARIABLE equals_err
)
if(equals_rc EQUAL 0)
    message(FATAL_ERROR "--model=--bogus unexpectedly succeeded")
endif()
if(NOT equals_rc EQUAL 1)
    message(FATAL_ERROR "--model=--bogus should reach model loading and exit 1, got ${equals_rc}: ${equals_err}")
endif()
if(NOT equals_err MATCHES "model file not found: --bogus")
    message(FATAL_ERROR "leading-dash value was not passed as a path; stderr: ${equals_err}")
endif()
if(NOT equals_out STREQUAL "")
    message(FATAL_ERROR "--model=--bogus wrote stdout: ${equals_out}")
endif()

# 4. Unexpected positional argument must also exit non-zero.
execute_process(
    COMMAND "${BIN}" unexpected-positional
    RESULT_VARIABLE pos_rc
    OUTPUT_VARIABLE pos_out
    ERROR_VARIABLE pos_err
)
if(pos_rc EQUAL 0)
    message(FATAL_ERROR "unexpected positional argument unexpectedly exited with 0")
endif()
if(NOT pos_err MATCHES "unexpected argument")
    message(FATAL_ERROR "positional-argument stderr is missing 'unexpected argument'; got: ${pos_err}")
endif()

# 5. No arguments must exit 0 and print the startup banner.
execute_process(
    COMMAND "${BIN}"
    RESULT_VARIABLE default_rc
    OUTPUT_VARIABLE default_out
    ERROR_VARIABLE default_err
)
if(NOT default_rc EQUAL 0)
    message(FATAL_ERROR "no-argument run exited with ${default_rc}; stderr: ${default_err}")
endif()
if(NOT default_out MATCHES "Industrial Defect Monitor System")
    message(FATAL_ERROR "no-argument run is missing the startup banner; got: ${default_out}")
endif()

# 6. Single-image input errors are independent of the real model.
if(NOT DEFINED TEST_DIR)
    message(FATAL_ERROR "TEST_DIR is not defined")
endif()
file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}")
file(WRITE "${TEST_DIR}/corrupt.jpg" "not an image")

function(expect_cli_failure label expected_regex)
    execute_process(
        COMMAND "${BIN}" ${ARGN}
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
    )
    if(rc EQUAL 0)
        message(FATAL_ERROR "${label}: expected non-zero exit, got 0")
    endif()
    if(NOT err MATCHES "${expected_regex}")
        message(FATAL_ERROR "${label}: stderr does not match '${expected_regex}'; got: ${err}")
    endif()
    if(NOT out STREQUAL "")
        message(FATAL_ERROR "${label}: expected empty stdout, got: ${out}")
    endif()
endfunction()

expect_cli_failure(
    "missing image"
    "image file not found"
    --model "${FIXTURES}/zero_output.onnx"
    --image "${TEST_DIR}/does-not-exist.png"
)
expect_cli_failure(
    "unsupported image extension"
    "unsupported image extension"
    --model "${FIXTURES}/zero_output.onnx"
    --image "${FIXTURES}/corrupt_model.bin"
)
expect_cli_failure(
    "corrupt image"
    "failed to read image"
    --model "${FIXTURES}/zero_output.onnx"
    --image "${TEST_DIR}/corrupt.jpg"
)

# 7. Invalid confidence/IoU thresholds must be rejected explicitly (exit 2).
function(expect_invalid_threshold label expected_regex)
    execute_process(
        COMMAND "${BIN}"
            --model "${FIXTURES}/zero_output.onnx"
            --image "${FIXTURES}/non_square_320x200.png"
            ${ARGN}
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
    )
    if(NOT rc EQUAL 2)
        message(FATAL_ERROR "${label}: expected exit 2, got ${rc}: ${err}")
    endif()
    if(NOT err MATCHES "${expected_regex}")
        message(FATAL_ERROR "${label}: stderr does not match '${expected_regex}'; got: ${err}")
    endif()
    if(NOT out STREQUAL "")
        message(FATAL_ERROR "${label}: expected empty stdout, got: ${out}")
    endif()
endfunction()

expect_invalid_threshold("conf 0" "must be in" --conf 0)
expect_invalid_threshold("conf 1.1" "must be in" --conf 1.1)
expect_invalid_threshold("conf nan" "must be in" --conf=nan)
expect_invalid_threshold("conf inf" "must be in" --conf=inf)
expect_invalid_threshold("iou 0" "must be in" --iou 0)
expect_invalid_threshold("iou 1.1" "must be in" --iou 1.1)

# 7b. --queue-capacity validation.
execute_process(
    COMMAND "${BIN}" --model "${FIXTURES}/zero_output.onnx"
        --image "${FIXTURES}/non_square_320x200.png"
        --queue-capacity 1
    RESULT_VARIABLE queue_single_rc
    OUTPUT_VARIABLE queue_single_out
    ERROR_VARIABLE queue_single_err
)
if(NOT queue_single_rc EQUAL 2 OR
   NOT queue_single_err MATCHES "only supported with --image-dir")
    message(FATAL_ERROR
        "--queue-capacity with single image not rejected: rc=${queue_single_rc}, stderr=${queue_single_err}")
endif()

execute_process(
    COMMAND "${BIN}" --model "${FIXTURES}/zero_output.onnx"
        --image-dir "${FIXTURES}"
        --queue-capacity 0
    RESULT_VARIABLE queue_zero_rc
    OUTPUT_VARIABLE queue_zero_out
    ERROR_VARIABLE queue_zero_err
)
if(NOT queue_zero_rc EQUAL 2 OR
   NOT queue_zero_err MATCHES "greater than 0")
    message(FATAL_ERROR
        "--queue-capacity 0 not rejected: rc=${queue_zero_rc}, stderr=${queue_zero_err}")
endif()

expect_invalid_threshold(
    "queue capacity -1" "greater than 0" --queue-capacity=-1
)
expect_invalid_threshold(
    "queue capacity overflow" "invalid value"
    --queue-capacity=18446744073709551616
)

# 8. A valid image with a zero-output model must report OK.
execute_process(
    COMMAND "${BIN}" --model "${FIXTURES}/zero_output.onnx"
        --image "${FIXTURES}/non_square_320x200.png"
    RESULT_VARIABLE ok_rc
    OUTPUT_VARIABLE ok_out
    ERROR_VARIABLE ok_err
)
if(NOT ok_rc EQUAL 0)
    message(FATAL_ERROR "single OK run exited ${ok_rc}; stderr: ${ok_err}")
endif()
string(FIND "${ok_out}" "quality: OK" ok_pos)
if(ok_pos EQUAL -1)
    message(FATAL_ERROR "single OK run did not report quality: OK; stdout: ${ok_out}")
endif()

message(STATUS "All edge_ai CLI checks passed")

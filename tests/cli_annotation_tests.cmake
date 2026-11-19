# Headless annotation-save and option-guard tests. No desktop/GUI is needed.
#
# Invoked by CTest:
#   cmake -DBIN=<edge_ai> -DFIXTURES=<dir> -DTEST_DIR=<dir> -P ...

if(NOT DEFINED BIN OR NOT EXISTS "${BIN}")
    message(FATAL_ERROR "edge_ai executable not found: ${BIN}")
endif()
if(NOT DEFINED FIXTURES OR NOT EXISTS
        "${FIXTURES}/non_square_320x200.png")
    message(FATAL_ERROR "fixtures not found: ${FIXTURES}")
endif()

file(MAKE_DIRECTORY "${TEST_DIR}")
set(OUT "${TEST_DIR}/annotated.png")

execute_process(
    COMMAND "${BIN}"
        --model "${FIXTURES}/zero_output.onnx"
        --image "${FIXTURES}/non_square_320x200.png"
        --save-annotated "${OUT}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR
        "headless annotated save failed: rc=${rc}; stdout=${out}; stderr=${err}")
endif()
if(NOT EXISTS "${OUT}")
    message(FATAL_ERROR "annotated image was not written: ${OUT}")
endif()
file(SIZE "${OUT}" size)
if(size LESS 100)
    message(FATAL_ERROR "annotated image is suspiciously small: ${size} bytes")
endif()

# Verify the saved header preserves the non-square dimensions.
execute_process(
    COMMAND file "${OUT}"
    OUTPUT_VARIABLE file_out
    ERROR_VARIABLE file_err)
if(NOT file_out MATCHES "320 x 200")
    message(FATAL_ERROR
        "annotated image lost its dimensions: ${file_out}${file_err}")
endif()

function(expect_option_error label expected_regex)
    execute_process(
        COMMAND "${BIN}" ${ARGN}
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err)
    if(NOT rc EQUAL 2)
        message(FATAL_ERROR
            "${label}: expected exit 2, got ${rc}; stdout=${out}; stderr=${err}")
    endif()
    if(NOT err MATCHES "${expected_regex}")
        message(FATAL_ERROR
            "${label}: stderr does not match '${expected_regex}'; got: ${err}")
    endif()
endfunction()

expect_option_error(
    "annotated output conflicts with input"
    "conflicts with input image"
    --model "${FIXTURES}/zero_output.onnx"
    --image "${FIXTURES}/non_square_320x200.png"
    --save-annotated "${FIXTURES}/non_square_320x200.png")

expect_option_error(
    "annotated output conflicts with json output"
    "refer to the same file"
    --model "${FIXTURES}/zero_output.onnx"
    --image "${FIXTURES}/non_square_320x200.png"
    --save-annotated "${TEST_DIR}/same.out"
    --json-out "${TEST_DIR}/same.out")

expect_option_error(
    "save-annotated requires a single image"
    "only supported with --image"
    --model "${FIXTURES}/zero_output.onnx"
    --image-dir "${FIXTURES}"
    --save-annotated "${TEST_DIR}/batch.png")

expect_option_error(
    "display rejects image directories"
    "not supported with --image-dir"
    --model "${FIXTURES}/zero_output.onnx"
    --image-dir "${FIXTURES}"
    --display)

expect_option_error(
    "display requires an image or camera"
    "requires --image or --camera"
    --display)

# In a headless shell --display must fail cleanly instead of letting Qt abort
# the process. QT_QPA_PLATFORM=offscreen remains an explicit opt-in.
execute_process(
    COMMAND env -u DISPLAY -u WAYLAND_DISPLAY -u QT_QPA_PLATFORM
        "${BIN}"
        --model "${FIXTURES}/zero_output.onnx"
        --image "${FIXTURES}/non_square_320x200.png"
        --display
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
if(NOT rc EQUAL 1)
    message(FATAL_ERROR
        "headless --display: expected exit 1, got ${rc}; stdout=${out}; stderr=${err}")
endif()
if(NOT err MATCHES "requires DISPLAY")
    message(FATAL_ERROR
        "headless --display did not explain the missing display: ${err}")
endif()

message(STATUS "Headless annotation save and option guards passed")

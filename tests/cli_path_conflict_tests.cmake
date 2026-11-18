# Verify that output paths cannot overwrite the model, the input image, or
# each other. Covers relative aliases, symlinks and hard links, and checks that
# the original files are byte-identical afterwards.
#
# Invoked by CTest:
#   cmake -DBIN=<edge_ai> -DFIXTURES=<tests/fixtures> -DTEST_DIR=<dir> \
#         -P cli_path_conflict_tests.cmake

if(NOT DEFINED BIN OR NOT EXISTS "${BIN}")
    message(FATAL_ERROR "edge_ai executable not found: ${BIN}")
endif()
if(NOT DEFINED FIXTURES OR NOT EXISTS "${FIXTURES}/zero_output.onnx")
    message(FATAL_ERROR "FIXTURES is not set or missing zero_output.onnx: ${FIXTURES}")
endif()
if(NOT DEFINED TEST_DIR)
    message(FATAL_ERROR "TEST_DIR is not defined")
endif()

file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}")
file(COPY "${FIXTURES}/zero_output.onnx" DESTINATION "${TEST_DIR}")
file(COPY "${FIXTURES}/non_square_320x200.png" DESTINATION "${TEST_DIR}")
set(MODEL "${TEST_DIR}/zero_output.onnx")
set(IMAGE "${TEST_DIR}/non_square_320x200.png")
file(SHA256 "${MODEL}" MODEL_SHA_BEFORE)
file(SHA256 "${IMAGE}" IMAGE_SHA_BEFORE)

function(expect_conflict label expected_regex)
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

expect_conflict(
    "json-out vs model"
    "conflicts with model"
    --model "${MODEL}" --image "${IMAGE}" --json-out "${MODEL}"
)
expect_conflict(
    "json-out vs image"
    "conflicts with input image"
    --model "${MODEL}" --image "${IMAGE}" --json-out "${IMAGE}"
)
expect_conflict(
    "dump-input vs model"
    "conflicts with model"
    --model "${MODEL}" --image "${IMAGE}" --dump-input "${MODEL}"
)
expect_conflict(
    "dump-input vs image"
    "conflicts with input image"
    --model "${MODEL}" --image "${IMAGE}" --dump-input "${IMAGE}"
)

set(SAME_OUT "${TEST_DIR}/same-output.bin")
expect_conflict(
    "same output path"
    "refer to the same file"
    --model "${MODEL}" --image "${IMAGE}"
    --json-out "${SAME_OUT}" --dump-input "${SAME_OUT}"
)
if(EXISTS "${SAME_OUT}")
    message(FATAL_ERROR "conflicting output path was created: ${SAME_OUT}")
endif()

# Two textual spellings of the same not-yet-created output path.
execute_process(
    COMMAND "${BIN}" --model zero_output.onnx --image non_square_320x200.png
            --json-out same-relative.bin --dump-input ./same-relative.bin
    WORKING_DIRECTORY "${TEST_DIR}"
    RESULT_VARIABLE same_relative_rc
    OUTPUT_VARIABLE same_relative_out
    ERROR_VARIABLE same_relative_err
)
if(same_relative_rc EQUAL 0 OR NOT same_relative_err MATCHES "refer to the same file")
    message(FATAL_ERROR
        "relative same-output alias not rejected: rc=${same_relative_rc}, stderr=${same_relative_err}")
endif()
if(EXISTS "${TEST_DIR}/same-relative.bin")
    message(FATAL_ERROR "relative conflicting output path was created")
endif()
message(STATUS "relative same-output alias: rc=${same_relative_rc}; stderr: ${same_relative_err}")

# Relative alias for the model path.
execute_process(
    COMMAND "${BIN}" --model zero_output.onnx --image non_square_320x200.png
            --json-out ./zero_output.onnx
    WORKING_DIRECTORY "${TEST_DIR}"
    RESULT_VARIABLE relative_rc
    OUTPUT_VARIABLE relative_out
    ERROR_VARIABLE relative_err
)
if(relative_rc EQUAL 0 OR NOT relative_err MATCHES "conflicts with model")
    message(FATAL_ERROR
        "relative alias not rejected: rc=${relative_rc}, stderr=${relative_err}")
endif()
message(STATUS "relative alias: rc=${relative_rc}; stderr: ${relative_err}")

# Symlink alias.
file(CREATE_LINK "${MODEL}" "${TEST_DIR}/model-symlink.onnx" SYMBOLIC)
expect_conflict(
    "symlink to model"
    "conflicts with model"
    --model "${MODEL}" --image "${IMAGE}"
    --json-out "${TEST_DIR}/model-symlink.onnx"
)

# Hard-link alias (requires both paths to exist, which exercises equivalent()).
# file(CREATE_LINK) creates a hard link by default.
file(CREATE_LINK "${MODEL}" "${TEST_DIR}/model-hardlink.onnx")
expect_conflict(
    "hardlink to model"
    "conflicts with model"
    --model "${MODEL}" --image "${IMAGE}"
    --json-out "${TEST_DIR}/model-hardlink.onnx"
)

# ".." alias.
file(MAKE_DIRECTORY "${TEST_DIR}/sub")
expect_conflict(
    "dot-dot alias to model"
    "conflicts with model"
    --model "${MODEL}" --image "${IMAGE}"
    --json-out "${TEST_DIR}/sub/../zero_output.onnx"
)

# Dangling symlink whose target is the other output path. Writing the Tensor
# through the alias would create result.json, then the JSON write would
# overwrite it; the program must reject the pair before creating the target.
set(DANGLING_TARGET "${TEST_DIR}/dangling-result.json")
set(DANGLING_LINK "${TEST_DIR}/dangling-alias.json")
file(CREATE_LINK "${DANGLING_TARGET}" "${DANGLING_LINK}" SYMBOLIC)
expect_conflict(
    "dangling symlink vs other output"
    "refer to the same file"
    --model "${MODEL}" --image "${IMAGE}"
    --dump-input "${DANGLING_LINK}" --json-out "${DANGLING_TARGET}"
)
if(EXISTS "${DANGLING_TARGET}")
    message(FATAL_ERROR "dangling symlink target was created: ${DANGLING_TARGET}")
endif()
# EXISTS() follows symlinks, so a dangling link reports false; READ_SYMLINK
# verifies that the link itself is still present.
file(READ_SYMLINK "${DANGLING_LINK}" DANGLING_LINK_VALUE)

# The same dangling target through a symlink chain.
set(CHAINED_LINK "${TEST_DIR}/chained-alias.json")
file(CREATE_LINK "${DANGLING_LINK}" "${CHAINED_LINK}" SYMBOLIC)
expect_conflict(
    "chained dangling symlink vs other output"
    "refer to the same file"
    --model "${MODEL}" --image "${IMAGE}"
    --json-out "${DANGLING_TARGET}" --dump-input "${CHAINED_LINK}"
)
if(EXISTS "${DANGLING_TARGET}")
    message(FATAL_ERROR "chained dangling symlink target was created: ${DANGLING_TARGET}")
endif()

file(SHA256 "${MODEL}" MODEL_SHA_AFTER)
file(SHA256 "${IMAGE}" IMAGE_SHA_AFTER)
if(NOT MODEL_SHA_BEFORE STREQUAL MODEL_SHA_AFTER)
    message(FATAL_ERROR "model file was modified despite path-conflict checks")
endif()
if(NOT IMAGE_SHA_BEFORE STREQUAL IMAGE_SHA_AFTER)
    message(FATAL_ERROR "image file was modified despite path-conflict checks")
endif()

message(STATUS "All output path-conflict checks passed; model/image hashes unchanged")

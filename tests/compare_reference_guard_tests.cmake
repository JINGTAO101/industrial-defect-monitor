# Negative tests for tools/compare_cpp_reference.py's completeness guards:
#   1. a directory with only one of the required six images must fail;
#   2. thresholds different from the reference results must fail.
#
# Registered by CTest only when the real model and image directory are provided.

foreach(var PYTHON EDGE_AI MODEL IMAGE_DIR REFERENCE_RESULTS SCRIPT TEST_DIR)
    if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
        message(FATAL_ERROR "${var} is not defined")
    endif()
endforeach()

file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}/one-image")
file(COPY "${IMAGE_DIR}/crazing_101.jpg" DESTINATION "${TEST_DIR}/one-image")

execute_process(
    COMMAND "${PYTHON}" -I "${SCRIPT}"
        --edge-ai "${EDGE_AI}"
        --model "${MODEL}"
        --image-dir "${TEST_DIR}/one-image"
        --reference-results "${REFERENCE_RESULTS}"
        --out-dir "${TEST_DIR}/out-missing"
    RESULT_VARIABLE missing_rc
    OUTPUT_VARIABLE missing_out
    ERROR_VARIABLE missing_err
)
if(missing_rc EQUAL 0)
    message(FATAL_ERROR "one-image comparison unexpectedly succeeded")
endif()
if(NOT missing_err MATCHES "image set mismatch")
    message(FATAL_ERROR "missing-image guard did not report 'image set mismatch'; stderr: ${missing_err}")
endif()
if(NOT missing_err MATCHES "missing")
    message(FATAL_ERROR "missing-image guard did not name missing images; stderr: ${missing_err}")
endif()
message(STATUS "one-image guard: rc=${missing_rc}; stderr: ${missing_err}")

execute_process(
    COMMAND "${PYTHON}" -I "${SCRIPT}"
        --edge-ai "${EDGE_AI}"
        --model "${MODEL}"
        --image-dir "${IMAGE_DIR}"
        --reference-results "${REFERENCE_RESULTS}"
        --out-dir "${TEST_DIR}/out-threshold"
        --conf 0.5
    RESULT_VARIABLE threshold_rc
    OUTPUT_VARIABLE threshold_out
    ERROR_VARIABLE threshold_err
)
if(threshold_rc EQUAL 0)
    message(FATAL_ERROR "threshold-mismatch comparison unexpectedly succeeded")
endif()
if(NOT threshold_err MATCHES "do not match the reference results")
    message(FATAL_ERROR "threshold guard did not report a mismatch; stderr: ${threshold_err}")
endif()
message(STATUS "threshold guard: rc=${threshold_rc}; stderr: ${threshold_err}")

message(STATUS "Reference comparison guard checks passed")

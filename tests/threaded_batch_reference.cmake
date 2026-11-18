# End-to-end threaded --image-dir run against the six reference images.
# Registered only when the real model and image directory are available.

foreach(var BIN PYTHON MODEL IMAGE_DIR REFERENCE_RESULTS COMPARE_SCRIPT TEST_DIR)
    if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
        message(FATAL_ERROR "${var} is not defined")
    endif()
endforeach()

file(GLOB REAL_IMAGES "${IMAGE_DIR}/*.jpg")
list(LENGTH REAL_IMAGES real_image_count)
if(NOT real_image_count EQUAL 6)
    message(FATAL_ERROR
        "expected six reference images, found ${real_image_count} in ${IMAGE_DIR}")
endif()
file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}/images")
file(COPY ${REAL_IMAGES} DESTINATION "${TEST_DIR}/images")

execute_process(
    COMMAND "${BIN}" --model "${MODEL}" --image-dir "${TEST_DIR}/images"
        --queue-capacity 1 --conf 0.25 --iou 0.45
        --json-out "${TEST_DIR}/batch.json"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "threaded batch exited ${rc}; stderr: ${err}")
endif()
foreach(expected
        "threads: reader="
        "queues: input capacity=1"
        "summary: total=6 ok=0 ng=6 errors=0")
    string(FIND "${out}" "${expected}" found)
    if(found EQUAL -1)
        message(FATAL_ERROR "batch output missing '${expected}': ${out}")
    endif()
endforeach()

execute_process(
    COMMAND "${PYTHON}" -I "${COMPARE_SCRIPT}"
        --batch-json "${TEST_DIR}/batch.json"
        --reference-results "${REFERENCE_RESULTS}"
        --conf 0.25 --iou 0.45
    RESULT_VARIABLE compare_rc
    OUTPUT_VARIABLE compare_out
    ERROR_VARIABLE compare_err
)
if(NOT compare_rc EQUAL 0)
    message(FATAL_ERROR
        "threaded batch comparison failed (${compare_rc}): ${compare_out}${compare_err}")
endif()
message(STATUS "${compare_out}")
message(STATUS "Threaded six-image batch matches the reference order and detections")

# Batch input, OK/NG reporting and bad-file behavior.
# Invoked by CTest:
#   cmake -DBIN=<edge_ai> -DFIXTURES=<tests/fixtures> -DTEST_DIR=<dir>
#         [-DREAL_MODEL=<model.onnx> -DREAL_IMAGE_DIR=<six-image dir>]
#         -P cli_batch_tests.cmake

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
file(MAKE_DIRECTORY "${TEST_DIR}/valid" "${TEST_DIR}/mixed")
file(COPY_FILE "${FIXTURES}/non_square_320x200.png"
     "${TEST_DIR}/valid/valid.png")
file(COPY_FILE "${FIXTURES}/non_square_320x200.png"
     "${TEST_DIR}/mixed/valid.png")
file(WRITE "${TEST_DIR}/mixed/bad.jpg" "not an image")
file(WRITE "${TEST_DIR}/mixed/note.txt" "unsupported file")

# 1. All-valid batch: exit 0, OK, no errors.
execute_process(
    COMMAND "${BIN}" --model "${FIXTURES}/zero_output.onnx"
        --image-dir "${TEST_DIR}/valid"
        --conf 0.25 --iou 0.45
        --json-out "${TEST_DIR}/valid.json"
    RESULT_VARIABLE valid_rc
    OUTPUT_VARIABLE valid_out
    ERROR_VARIABLE valid_err
)
if(NOT valid_rc EQUAL 0)
    message(FATAL_ERROR "valid batch exited ${valid_rc}; stderr: ${valid_err}")
endif()
foreach(expected "[OK]" "summary: total=1 ok=1 ng=0 errors=0 skipped_unsupported=0")
    string(FIND "${valid_out}" "${expected}" found)
    if(found EQUAL -1)
        message(FATAL_ERROR "valid batch output missing '${expected}': ${valid_out}")
    endif()
endforeach()
file(READ "${TEST_DIR}/valid.json" valid_json)
foreach(expected "\"ng\": 0" "\"errors\": 0" "\"skipped_unsupported\": 0")
    string(FIND "${valid_json}" "${expected}" found)
    if(found EQUAL -1)
        message(FATAL_ERROR "valid batch JSON missing '${expected}'")
    endif()
endforeach()
string(FIND "${valid_json}" "\"timestamp_unix_ms\":" timestamp_pos)
if(timestamp_pos EQUAL -1)
    message(FATAL_ERROR "valid batch JSON has no frame timestamp")
endif()
string(FIND "${valid_json}" "\"timestamp_unix_ms\": null" null_timestamp_pos)
if(NOT null_timestamp_pos EQUAL -1)
    message(FATAL_ERROR "valid batch JSON has a null timestamp")
endif()

# 2. Mixed batch: corrupt image is reported, processing continues, exit 1,
#    and an unsupported extension is counted as skipped.
execute_process(
    COMMAND "${BIN}" --model "${FIXTURES}/zero_output.onnx"
        --image-dir "${TEST_DIR}/mixed"
        --conf 0.25 --iou 0.45
        --json-out "${TEST_DIR}/mixed.json"
    RESULT_VARIABLE mixed_rc
    OUTPUT_VARIABLE mixed_out
    ERROR_VARIABLE mixed_err
)
if(mixed_rc EQUAL 0)
    message(FATAL_ERROR "mixed batch with a corrupt image must exit non-zero")
endif()
foreach(expected "[OK]" "[ERROR]" "summary: total=2 ok=1 ng=0 errors=1 skipped_unsupported=1")
    string(FIND "${mixed_out}" "${expected}" found)
    if(found EQUAL -1)
        message(FATAL_ERROR "mixed batch output missing '${expected}': ${mixed_out}")
    endif()
endforeach()
file(READ "${TEST_DIR}/mixed.json" mixed_json)
foreach(expected "\"errors\": 1" "\"skipped_unsupported\": 1")
    string(FIND "${mixed_json}" "${expected}" found)
    if(found EQUAL -1)
        message(FATAL_ERROR "mixed batch JSON missing '${expected}'")
    endif()
endforeach()
# The failed frame must not publish is_ng:false; only the completed OK frame
# may contain a boolean quality decision.
string(FIND "${mixed_json}" "\"quality\": null" null_quality_pos)
if(null_quality_pos EQUAL -1)
    message(FATAL_ERROR "failed frame does not set quality to null")
endif()
string(REGEX MATCHALL "\"is_ng\": false" mixed_false "${mixed_json}")
list(LENGTH mixed_false mixed_false_count)
if(NOT mixed_false_count EQUAL 1)
    message(FATAL_ERROR
        "expected exactly one completed OK frame with is_ng:false, got ${mixed_false_count}")
endif()

# 3. A batch containing only a corrupt image must expose quality:null and no
#    is_ng boolean at all.
file(MAKE_DIRECTORY "${TEST_DIR}/bad-only")
file(WRITE "${TEST_DIR}/bad-only/bad.jpg" "not an image")
execute_process(
    COMMAND "${BIN}" --model "${FIXTURES}/zero_output.onnx"
        --image-dir "${TEST_DIR}/bad-only"
        --json-out "${TEST_DIR}/bad-only.json"
    RESULT_VARIABLE bad_only_rc
    OUTPUT_VARIABLE bad_only_out
    ERROR_VARIABLE bad_only_err
)
if(bad_only_rc EQUAL 0)
    message(FATAL_ERROR "bad-only batch must exit non-zero")
endif()
file(READ "${TEST_DIR}/bad-only.json" bad_only_json)
string(FIND "${bad_only_json}" "\"quality\": null" bad_null_pos)
if(bad_null_pos EQUAL -1)
    message(FATAL_ERROR "bad-only batch does not set quality to null")
endif()
string(FIND "${bad_only_json}" "\"is_ng\"" bad_is_ng_pos)
if(NOT bad_is_ng_pos EQUAL -1)
    message(FATAL_ERROR "bad-only batch published an is_ng boolean for a failed frame")
endif()

# 3. --json-out must not overwrite a batch input image.
file(SHA256 "${TEST_DIR}/valid/valid.png" conflict_sha_before)
execute_process(
    COMMAND "${BIN}" --model "${FIXTURES}/zero_output.onnx"
        --image-dir "${TEST_DIR}/valid"
        --json-out "${TEST_DIR}/valid/valid.png"
    RESULT_VARIABLE conflict_rc
    OUTPUT_VARIABLE conflict_out
    ERROR_VARIABLE conflict_err
)
if(conflict_rc EQUAL 0 OR NOT conflict_err MATCHES "conflicts with input image")
    message(FATAL_ERROR
        "batch output conflict not rejected: rc=${conflict_rc}, stderr=${conflict_err}")
endif()
file(SHA256 "${TEST_DIR}/valid/valid.png" conflict_sha_after)
if(NOT conflict_sha_before STREQUAL conflict_sha_after)
    message(FATAL_ERROR "batch input image was modified by conflicting output")
endif()

# 4. --dump-input is single-image only.
execute_process(
    COMMAND "${BIN}" --model "${FIXTURES}/zero_output.onnx"
        --image-dir "${TEST_DIR}/valid"
        --dump-input "${TEST_DIR}/should-not-exist.bin"
    RESULT_VARIABLE dump_rc
    OUTPUT_VARIABLE dump_out
    ERROR_VARIABLE dump_err
)
if(dump_rc EQUAL 0 OR NOT dump_err MATCHES "only supported with --image")
    message(FATAL_ERROR
        "batch --dump-input not rejected: rc=${dump_rc}, stderr=${dump_err}")
endif()
if(EXISTS "${TEST_DIR}/should-not-exist.bin")
    message(FATAL_ERROR "batch --dump-input created a file")
endif()

# 5. --image and --image-dir are mutually exclusive.
execute_process(
    COMMAND "${BIN}" --model "${FIXTURES}/zero_output.onnx"
        --image "${FIXTURES}/non_square_320x200.png"
        --image-dir "${TEST_DIR}/valid"
    RESULT_VARIABLE both_rc
    OUTPUT_VARIABLE both_out
    ERROR_VARIABLE both_err
)
if(both_rc EQUAL 0 OR NOT both_err MATCHES "mutually exclusive")
    message(FATAL_ERROR
        "single/batch input combination not rejected: rc=${both_rc}, stderr=${both_err}")
endif()

# 6. Optional real-model batch: all six NEU-DET images contain detections at
#    conf 0.25, so the batch must report NG for all of them and exit 0.
if(DEFINED REAL_MODEL AND NOT REAL_MODEL STREQUAL "" AND
   DEFINED REAL_IMAGE_DIR AND NOT REAL_IMAGE_DIR STREQUAL "")
    file(GLOB REAL_IMAGES "${REAL_IMAGE_DIR}/*.jpg")
    list(LENGTH REAL_IMAGES real_image_count)
    if(NOT real_image_count EQUAL 6)
        message(FATAL_ERROR
            "expected six real-model images, found ${real_image_count} in ${REAL_IMAGE_DIR}")
    endif()
    file(MAKE_DIRECTORY "${TEST_DIR}/real-batch")
    file(COPY ${REAL_IMAGES} DESTINATION "${TEST_DIR}/real-batch")
    execute_process(
        COMMAND "${BIN}" --model "${REAL_MODEL}"
            --image-dir "${TEST_DIR}/real-batch"
            --conf 0.25 --iou 0.45
            --json-out "${TEST_DIR}/real-batch.json"
        RESULT_VARIABLE real_rc
        OUTPUT_VARIABLE real_out
        ERROR_VARIABLE real_err
    )
    if(NOT real_rc EQUAL 0)
        message(FATAL_ERROR "real batch exited ${real_rc}; stderr: ${real_err}")
    endif()
    string(FIND "${real_out}"
           "summary: total=6 ok=0 ng=6 errors=0 skipped_unsupported=0" found)
    if(found EQUAL -1)
        message(FATAL_ERROR "real batch summary is unexpected: ${real_out}")
    endif()
    file(READ "${TEST_DIR}/real-batch.json" real_json)
    string(FIND "${real_json}" "\"ng\": 6" found)
    if(found EQUAL -1)
        message(FATAL_ERROR "real batch JSON does not report ng=6")
    endif()
    message(STATUS "real-model batch: all six images classified NG")
else()
    message(STATUS "real-model batch: SKIPPED (REAL_MODEL/REAL_IMAGE_DIR not provided)")
endif()

message(STATUS "All batch input and OK/NG checks passed")

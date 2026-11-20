# NG image retention CLI acceptance. Uses synthetic fixtures only.
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
set(DB "${TEST_DIR}/ng-events.db")
set(NG_ORIGINAL "${TEST_DIR}/ng-original")
set(NG_ANNOTATED "${TEST_DIR}/ng-annotated")
set(NG_OK "${TEST_DIR}/ng-ok")
set(NG_WRITE_FAIL "${TEST_DIR}/ng-write-fail")
set(NG_CLEANUP "${TEST_DIR}/ng-cleanup")
file(REMOVE "${DB}" "${DB}-journal" "${DB}-wal" "${DB}-shm")
file(REMOVE_RECURSE "${NG_ORIGINAL}" "${NG_ANNOTATED}" "${NG_OK}"
    "${NG_WRITE_FAIL}" "${NG_CLEANUP}")

function(run_edge out_var err_var rc_var)
    execute_process(
        COMMAND "${BIN}" ${ARGN}
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err)
    set(${out_var} "${out}" PARENT_SCOPE)
    set(${err_var} "${err}" PARENT_SCOPE)
    set(${rc_var} "${rc}" PARENT_SCOPE)
endfunction()

function(expect_rc label expected actual out err)
    if(NOT actual EQUAL expected)
        message(FATAL_ERROR
            "${label}: expected exit ${expected}, got ${actual}; "
            "stdout=${out}; stderr=${err}")
    endif()
endfunction()

function(query_product db product out_var)
    run_edge(out err rc --query --db "${db}" --product-id "${product}")
    expect_rc("query ${product}" 0 "${rc}" "${out}" "${err}")
    if(NOT out MATCHES "events: total=1")
        message(FATAL_ERROR "query ${product} count is wrong: ${out}")
    endif()
    set(${out_var} "${out}" PARENT_SCOPE)
endfunction()

function(extract_uid image_line uid_var image_var)
    string(REGEX MATCH "uid=([0-9a-f-]+)" uid_match "${image_line}")
    if(NOT uid_match)
        message(FATAL_ERROR "cannot find event uid: ${image_line}")
    endif()
    set(uid "${CMAKE_MATCH_1}")
    string(REGEX MATCH "image=([^ \n]+)" image_match "${image_line}")
    if(NOT image_match)
        message(FATAL_ERROR "cannot find image path: ${image_line}")
    endif()
    set(${uid_var} "${uid}" PARENT_SCOPE)
    set(${image_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()

# 1. Default off: NG event is stored, but image_path is NULL.
run_edge(out err rc
    --model "${FIXTURES}/fixed_detection.onnx"
    --image "${FIXTURES}/non_square_320x200.png"
    --product-id "NG-OFF" --db "${DB}")
expect_rc("default off write" 0 "${rc}" "${out}" "${err}")
query_product("${DB}" "NG-OFF" event_line)
if(NOT event_line MATCHES "image=-")
    message(FATAL_ERROR "default-off event unexpectedly has an image: ${event_line}")
endif()

# 2. original mode: byte-copy the input file under the event id.
run_edge(out err rc
    --model "${FIXTURES}/fixed_detection.onnx"
    --image "${FIXTURES}/non_square_320x200.png"
    --product-id "NG-ORIG" --db "${DB}"
    --ng-image-dir "${NG_ORIGINAL}" --ng-image-mode original)
expect_rc("original retention write" 0 "${rc}" "${out}" "${err}")
query_product("${DB}" "NG-ORIG" event_line)
extract_uid("${event_line}" uid image)
if(NOT EXISTS "${image}")
    message(FATAL_ERROR "query image path does not exist: ${image}")
endif()
get_filename_component(image_name "${image}" NAME)
string(FIND "${image_name}" "${uid}" prefix_pos)
if(NOT prefix_pos EQUAL 0)
    message(FATAL_ERROR "image '${image_name}' is not named after uid '${uid}'")
endif()
file(SHA256 "${FIXTURES}/non_square_320x200.png" input_sha)
file(SHA256 "${image}" image_sha)
if(NOT input_sha STREQUAL image_sha)
    message(FATAL_ERROR "original mode did not byte-copy the input")
endif()

# 3. annotated mode: write a PNG using the 5C annotation path.
run_edge(out err rc
    --model "${FIXTURES}/fixed_detection.onnx"
    --image "${FIXTURES}/non_square_320x200.png"
    --product-id "NG-ANN" --db "${DB}"
    --ng-image-dir "${NG_ANNOTATED}" --ng-image-mode annotated)
expect_rc("annotated retention write" 0 "${rc}" "${out}" "${err}")
query_product("${DB}" "NG-ANN" event_line)
extract_uid("${event_line}" uid image)
if(NOT EXISTS "${image}")
    message(FATAL_ERROR "annotated image path does not exist: ${image}")
endif()
get_filename_component(image_name "${image}" NAME)
string(FIND "${image_name}" "${uid}" prefix_pos)
if(NOT prefix_pos EQUAL 0)
    message(FATAL_ERROR "annotated image is not named after the event id")
endif()
execute_process(COMMAND file "${image}" OUTPUT_VARIABLE file_out)
if(NOT file_out MATCHES "PNG image data, 320 x 200")
    message(FATAL_ERROR "annotated image is not a 320x200 PNG: ${file_out}")
endif()

# 4. OK frames never produce a retained image.
run_edge(out err rc
    --model "${FIXTURES}/zero_output.onnx"
    --image "${FIXTURES}/non_square_320x200.png"
    --product-id "OK-NOIMAGE" --db "${DB}"
    --ng-image-dir "${NG_OK}" --ng-image-mode original)
expect_rc("OK retention write" 0 "${rc}" "${out}" "${err}")
query_product("${DB}" "OK-NOIMAGE" event_line)
if(NOT event_line MATCHES "status=OK" OR NOT event_line MATCHES "image=-")
    message(FATAL_ERROR "OK event unexpectedly has an image: ${event_line}")
endif()
file(GLOB ok_files "${NG_OK}/*")
list(LENGTH ok_files ok_count)
if(NOT ok_count EQUAL 0)
    message(FATAL_ERROR "OK frame created ${ok_count} retained image(s)")
endif()

# 5. Image write failure: NG event is still stored with image_path NULL,
#    the log identifies the event, and the CLI exits 1.
run_edge(out err rc
    --model "${FIXTURES}/fixed_detection.onnx"
    --image "${FIXTURES}/non_square_320x200.png"
    --product-id "NG-WRITE-FAIL" --db "${DB}"
    --ng-image-dir "${FIXTURES}/non_square_320x200.png/ng-images")
expect_rc("image write failure" 1 "${rc}" "${out}" "${err}")
if(NOT err MATCHES "NG image save failed" OR NOT err MATCHES "event uid")
    message(FATAL_ERROR "image write failure log is unclear: ${err}")
endif()
query_product("${DB}" "NG-WRITE-FAIL" event_line)
if(NOT event_line MATCHES "status=NG" OR NOT event_line MATCHES "image=-")
    message(FATAL_ERROR
        "NG event was not stored after image failure: ${event_line}")
endif()

# 6. DB failure after a successful image write: the newly written image is
#    removed again.
set(LEGACY_DB "${TEST_DIR}/legacy-ng.db")
file(REMOVE "${LEGACY_DB}" "${LEGACY_DB}-journal" "${LEGACY_DB}-wal"
    "${LEGACY_DB}-shm")
file(COPY_FILE "${FIXTURES}/legacy_no_meta.db" "${LEGACY_DB}")
run_edge(out err rc
    --model "${FIXTURES}/fixed_detection.onnx"
    --image "${FIXTURES}/non_square_320x200.png"
    --product-id "NG-CLEANUP" --db "${LEGACY_DB}"
    --ng-image-dir "${NG_CLEANUP}" --ng-image-mode annotated)
expect_rc("DB failure cleanup" 1 "${rc}" "${out}" "${err}")
if(NOT err MATCHES "event database open failed")
    message(FATAL_ERROR "DB failure log is unclear: ${err}")
endif()
if(NOT err MATCHES "removed retained image")
    message(FATAL_ERROR "DB failure did not log image cleanup: ${err}")
endif()
file(GLOB cleanup_files "${NG_CLEANUP}/*")
list(LENGTH cleanup_files cleanup_count)
if(NOT cleanup_count EQUAL 0)
    message(FATAL_ERROR
        "DB failure left ${cleanup_count} retained image(s) behind")
endif()

# 7. Option guards.
run_edge(out err rc
    --model "${FIXTURES}/fixed_detection.onnx"
    --image "${FIXTURES}/non_square_320x200.png"
    --ng-image-dir "${NG_OK}")
expect_rc("ng dir without db" 2 "${rc}" "${out}" "${err}")
if(NOT err MATCHES "requires --db")
    message(FATAL_ERROR "ng dir without db error is unclear: ${err}")
endif()

run_edge(out err rc
    --model "${FIXTURES}/fixed_detection.onnx"
    --image "${FIXTURES}/non_square_320x200.png"
    --db "${DB}" --ng-image-mode annotated)
expect_rc("ng mode without dir" 2 "${rc}" "${out}" "${err}")
if(NOT err MATCHES "requires --ng-image-dir")
    message(FATAL_ERROR "ng mode without dir error is unclear: ${err}")
endif()

run_edge(out err rc
    --model "${FIXTURES}/fixed_detection.onnx"
    --image "${FIXTURES}/non_square_320x200.png"
    --db "${DB}" --ng-image-dir "${NG_OK}" --ng-image-mode bad)
expect_rc("bad ng mode" 2 "${rc}" "${out}" "${err}")
if(NOT err MATCHES "invalid value for --ng-image-mode")
    message(FATAL_ERROR "bad ng mode error is unclear: ${err}")
endif()

message(STATUS "NG image retention CLI checks passed")

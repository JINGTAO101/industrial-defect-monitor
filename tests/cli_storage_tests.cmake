# SQLite event storage CLI acceptance test. Uses synthetic fixtures only.
#
# Invoked by CTest:
#   cmake -DBIN=<edge_ai> -DFIXTURES=<dir> -DTEST_DIR=<dir> -P ...

if(NOT DEFINED BIN OR NOT EXISTS "${BIN}")
    message(FATAL_ERROR "edge_ai executable not found: ${BIN}")
endif()
if(NOT DEFINED FIXTURES OR NOT EXISTS "${FIXTURES}/fixed_detection.onnx")
    message(FATAL_ERROR "fixtures not found: ${FIXTURES}")
endif()

file(MAKE_DIRECTORY "${TEST_DIR}")
set(DB "${TEST_DIR}/events.db")
file(REMOVE "${DB}" "${DB}-journal" "${DB}-wal" "${DB}-shm")

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

# 1. Write one OK and one NG event, separated in time so range filters can
#    select exactly one.
run_edge(out err rc
    --model "${FIXTURES}/zero_output.onnx"
    --image "${FIXTURES}/non_square_320x200.png"
    --product-id "PROD-OK"
    --db "${DB}")
expect_rc("write OK event" 0 "${rc}" "${out}" "${err}")
if(NOT out MATCHES "event saved:" OR NOT out MATCHES "quality: OK")
    message(FATAL_ERROR "OK event write output is wrong: ${out}")
endif()

execute_process(COMMAND sleep 1.1)
string(TIMESTAMP BOUND_SEC "%s")
math(EXPR BOUND_MS "${BOUND_SEC} * 1000")
execute_process(COMMAND sleep 1.1)

run_edge(out err rc
    --model "${FIXTURES}/fixed_detection.onnx"
    --image "${FIXTURES}/non_square_320x200.png"
    --product-id "PROD-NG"
    --db "${DB}")
expect_rc("write NG event" 0 "${rc}" "${out}" "${err}")
if(NOT out MATCHES "event saved:" OR NOT out MATCHES "quality: NG")
    message(FATAL_ERROR "NG event write output is wrong: ${out}")
endif()

# 2. Query from a fresh process: both events persist.
run_edge(out err rc --query --db "${DB}")
expect_rc("query all" 0 "${rc}" "${out}" "${err}")
if(NOT out MATCHES "events: total=2")
    message(FATAL_ERROR "query all count is wrong: ${out}")
endif()
foreach(needle
        "product=PROD-OK" "product=PROD-NG" "status=OK" "status=NG"
        "defect class_id=0 class=crazing confidence=0.900000")
    if(NOT out MATCHES "${needle}")
        message(FATAL_ERROR "query all missing '${needle}': ${out}")
    endif()
endforeach()

# 3. Individual and combined filters.
run_edge(out err rc --query --db "${DB}" --status NG)
expect_rc("filter NG" 0 "${rc}" "${out}" "${err}")
if(NOT out MATCHES "events: total=1" OR NOT out MATCHES "product=PROD-NG"
        OR out MATCHES "product=PROD-OK")
    message(FATAL_ERROR "NG filter is wrong: ${out}")
endif()

run_edge(out err rc --query --db "${DB}" --status OK)
expect_rc("filter OK" 0 "${rc}" "${out}" "${err}")
if(NOT out MATCHES "events: total=1" OR NOT out MATCHES "product=PROD-OK"
        OR out MATCHES "product=PROD-NG")
    message(FATAL_ERROR "OK filter is wrong: ${out}")
endif()

run_edge(out err rc --query --db "${DB}" --product-id "PROD-NG")
expect_rc("filter product" 0 "${rc}" "${out}" "${err}")
if(NOT out MATCHES "events: total=1" OR NOT out MATCHES "product=PROD-NG")
    message(FATAL_ERROR "product filter is wrong: ${out}")
endif()

run_edge(out err rc --query --db "${DB}" --to "${BOUND_MS}")
expect_rc("filter upper time bound" 0 "${rc}" "${out}" "${err}")
if(NOT out MATCHES "events: total=1" OR NOT out MATCHES "product=PROD-OK")
    message(FATAL_ERROR "upper time filter is wrong: ${out}")
endif()

run_edge(out err rc --query --db "${DB}" --from "${BOUND_MS}")
expect_rc("filter lower time bound" 0 "${rc}" "${out}" "${err}")
if(NOT out MATCHES "events: total=1" OR NOT out MATCHES "product=PROD-NG")
    message(FATAL_ERROR "lower time filter is wrong: ${out}")
endif()

run_edge(out err rc --query --db "${DB}" --status NG
    --product-id "PROD-NG" --from "${BOUND_MS}")
expect_rc("combined filters" 0 "${rc}" "${out}" "${err}")
if(NOT out MATCHES "events: total=1" OR NOT out MATCHES "product=PROD-NG")
    message(FATAL_ERROR "combined filters are wrong: ${out}")
endif()

run_edge(out err rc --query --db "${DB}" --from 9999999999999)
expect_rc("empty query result" 0 "${rc}" "${out}" "${err}")
if(NOT out MATCHES "events: total=0")
    message(FATAL_ERROR "empty query result is wrong: ${out}")
endif()

# 4. Database open failure is diagnostic and does not replace the quality
#    decision already printed by the detector.
run_edge(out err rc
    --model "${FIXTURES}/zero_output.onnx"
    --image "${FIXTURES}/non_square_320x200.png"
    --db "${FIXTURES}/non_square_320x200.png/events.db")
expect_rc("database open failure" 1 "${rc}" "${out}" "${err}")
if(NOT out MATCHES "quality: OK")
    message(FATAL_ERROR "quality output was lost on DB failure: ${out}")
endif()
if(NOT err MATCHES "database")
    message(FATAL_ERROR "database failure is not diagnostic: ${err}")
endif()

run_edge(out err rc --query --db "${TEST_DIR}/missing.db")
expect_rc("query missing database" 1 "${rc}" "${out}" "${err}")
if(NOT err MATCHES "cannot open database")
    message(FATAL_ERROR "missing database error is unclear: ${err}")
endif()

# 5. Option guards.
run_edge(out err rc --query)
expect_rc("query without db" 2 "${rc}" "${out}" "${err}")
if(NOT err MATCHES "requires --db")
    message(FATAL_ERROR "query without db error is unclear: ${err}")
endif()

run_edge(out err rc --query --db "${DB}"
    --image "${FIXTURES}/non_square_320x200.png")
expect_rc("query with image" 2 "${rc}" "${out}" "${err}")
if(NOT err MATCHES "cannot be combined")
    message(FATAL_ERROR "query with image error is unclear: ${err}")
endif()

run_edge(out err rc --db "${DB}" --image-dir "${FIXTURES}")
expect_rc("db with image-dir" 2 "${rc}" "${out}" "${err}")
if(NOT err MATCHES "single-image")
    message(FATAL_ERROR "db image-dir error is unclear: ${err}")
endif()

run_edge(out err rc --status BAD --query --db "${DB}")
expect_rc("invalid status" 2 "${rc}" "${out}" "${err}")
if(NOT err MATCHES "invalid value for --status")
    message(FATAL_ERROR "invalid status error is unclear: ${err}")
endif()

run_edge(out err rc --query --db "${DB}" --from not-a-time)
expect_rc("invalid timestamp" 2 "${rc}" "${out}" "${err}")
if(NOT err MATCHES "invalid value for --from")
    message(FATAL_ERROR "invalid timestamp error is unclear: ${err}")
endif()

# 6. SQLite sidecar paths must not be usable as output paths.
foreach(sidecar_suffix -journal -wal -shm)
    run_edge(out err rc
        --model "${FIXTURES}/zero_output.onnx"
        --image "${FIXTURES}/non_square_320x200.png"
        --db "${DB}"
        --json-out "${DB}${sidecar_suffix}")
    expect_rc("sidecar conflict ${sidecar_suffix}" 2 "${rc}" "${out}" "${err}")
    if(NOT err MATCHES "SQLite sidecar")
        message(FATAL_ERROR
            "sidecar conflict ${sidecar_suffix} is unclear: ${err}")
    endif()
endforeach()

# 7. In-memory databases cannot satisfy the persistence contract.
run_edge(out err rc
    --model "${FIXTURES}/zero_output.onnx"
    --image "${FIXTURES}/non_square_320x200.png"
    --db ":memory:")
expect_rc("in-memory database" 2 "${rc}" "${out}" "${err}")
if(NOT err MATCHES "persistent file path")
    message(FATAL_ERROR "in-memory database error is unclear: ${err}")
endif()

# 8. Impossible calendar dates are rejected, not normalized.
run_edge(out err rc --query --db "${DB}" --from 2026-02-31T00:00:00Z)
expect_rc("impossible calendar date" 2 "${rc}" "${out}" "${err}")
if(NOT err MATCHES "invalid value for --from")
    message(FATAL_ERROR "impossible date error is unclear: ${err}")
endif()

# 9. A failed optional export must not prevent the event from being saved.
set(EXPORT_DB "${TEST_DIR}/export-failure.db")
file(REMOVE "${EXPORT_DB}" "${EXPORT_DB}-journal" "${EXPORT_DB}-wal"
    "${EXPORT_DB}-shm")
run_edge(out err rc
    --model "${FIXTURES}/zero_output.onnx"
    --image "${FIXTURES}/non_square_320x200.png"
    --product-id "PROD-EXPORT"
    --db "${EXPORT_DB}"
    --save-annotated "${TEST_DIR}/annotated.unsupported")
expect_rc("annotation export failure" 1 "${rc}" "${out}" "${err}")
if(NOT err MATCHES "annotated image")
    message(FATAL_ERROR "annotation export error is unclear: ${err}")
endif()
run_edge(out err rc --query --db "${EXPORT_DB}")
expect_rc("query after export failure" 0 "${rc}" "${out}" "${err}")
if(NOT out MATCHES "events: total=1" OR NOT out MATCHES "product=PROD-EXPORT")
    message(FATAL_ERROR
        "event was not persisted before the failed export: ${out}")
endif()

message(STATUS "SQLite storage CLI checks passed")

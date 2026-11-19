# Image-mode SIGINT regression: Ctrl+C must terminate a batch with the default
# signal disposition, not be swallowed by a process-wide handler intended for
# camera modes.
#
# Invoked by CTest:
#   cmake -DBIN=<edge_ai> -DFIXTURES=<dir> -P cli_signal_tests.cmake

if(NOT DEFINED BIN OR NOT EXISTS "${BIN}")
    message(FATAL_ERROR "edge_ai executable not found: ${BIN}")
endif()
if(NOT DEFINED FIXTURES OR NOT EXISTS "${FIXTURES}/zero_output.onnx")
    message(FATAL_ERROR "fixtures not found: ${FIXTURES}")
endif()

if(NOT DEFINED TEST_DIR)
    set(TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/cli-signal-tests")
endif()
set(IMAGE_DIR "${TEST_DIR}/images")
file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${IMAGE_DIR}")

# Enough inputs that the batch is still running when SIGINT arrives.
execute_process(
    COMMAND bash -c
        "for i in $(seq -w 1 2000); do ln -f \"$1\" \"$2/$i.png\" || exit 1; done"
        _ "${FIXTURES}/non_square_320x200.png" "${IMAGE_DIR}"
    RESULT_VARIABLE LINK_RC
    OUTPUT_VARIABLE LINK_OUT
    ERROR_VARIABLE LINK_ERR)
if(NOT LINK_RC EQUAL 0)
    message(FATAL_ERROR
        "failed to prepare signal-test images: ${LINK_ERR}${LINK_OUT}")
endif()

# timeout runs the command in the foreground and sends SIGINT after 2s. This
# avoids the POSIX rule that non-interactive shells make background jobs
# ignore SIGINT, which would mask the application's disposition.
set(RUN_LOG "${TEST_DIR}/image-batch.log")
set(RUN_ERR_LOG "${TEST_DIR}/image-batch.err.log")
execute_process(
    COMMAND timeout --preserve-status -s INT -k 5s 2s
        stdbuf -oL "${BIN}" --model "${FIXTURES}/zero_output.onnx"
        --image-dir "${IMAGE_DIR}" --queue-capacity 1
    RESULT_VARIABLE RUN_RC
    OUTPUT_FILE "${RUN_LOG}"
    ERROR_FILE "${RUN_ERR_LOG}")

if(NOT RUN_RC EQUAL 130)
    file(READ "${RUN_LOG}" LOG_CONTENT LIMIT 4000)
    file(READ "${RUN_ERR_LOG}" ERR_CONTENT LIMIT 2000)
    message(FATAL_ERROR
        "image batch did not terminate on SIGINT: rc=${RUN_RC}; "
        "stderr=${ERR_CONTENT}; log=${LOG_CONTENT}")
endif()

file(READ "${RUN_LOG}" LOG_CONTENT)
if(LOG_CONTENT MATCHES "summary:")
    message(FATAL_ERROR
        "image batch completed after SIGINT instead of being interrupted")
endif()
if(NOT LOG_CONTENT MATCHES "\\[OK\\]")
    message(FATAL_ERROR "image batch did not start processing before SIGINT")
endif()

message(STATUS "Image-mode SIGINT terminates the batch (rc=${RUN_RC})")

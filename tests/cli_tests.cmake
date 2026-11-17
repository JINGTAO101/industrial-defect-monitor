# CLI behaviour tests for edge_ai, executed by CTest via `cmake -P`.
# Checks real process exit codes and stdout/stderr contents.

if(NOT DEFINED BIN)
    message(FATAL_ERROR "BIN is not defined (expected path to the edge_ai executable)")
endif()

if(NOT EXISTS "${BIN}")
    message(FATAL_ERROR "edge_ai executable not found: ${BIN}")
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

# 3. Unexpected positional argument must also exit non-zero.
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

# 4. No arguments must exit 0 and print the startup banner.
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

message(STATUS "All edge_ai CLI checks passed")

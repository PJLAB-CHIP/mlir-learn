# This is used for colorizing the output
# For example, to colorize the output with purple, use the following:
# > mlir_learn_log_info("${Esc}[0;35mHello, World!${Esc}[m")
string(ASCII 27 Esc)

set(MLIR_LEARN_LOG_PREFIX "MLIR-LEARN")

set(MLIR_LEARN_LOG_RED "${Esc}[0;31m")
set(MLIR_LEARN_LOG_GREEN "${Esc}[0;32m")
set(MLIR_LEARN_LOG_YELLOW "${Esc}[0;33m")
set(MLIR_LEARN_LOG_BLUE "${Esc}[0;34m")
set(MLIR_LEARN_LOG_PURPLE "${Esc}[0;35m")
set(MLIR_LEARN_LOG_CYAN "${Esc}[0;36m")
set(MLIR_LEARN_LOG_WHITE "${Esc}[0;37m")
set(MLIR_LEARN_LOG_RESET "${Esc}[m")

function(mlir_learn_log_info msg)
    message(STATUS "[${MLIR_LEARN_LOG_PREFIX}|${MLIR_LEARN_LOG_GREEN}INFO${MLIR_LEARN_LOG_RESET}] >>> ${msg}")
endfunction(mlir_learn_log_info msg)

function(log_warning msg)
    message(WARNING "[${MLIR_LEARN_LOG_PREFIX}|WARNING] >>> ${msg}")
endfunction(log_warning msg)

function(log_error msg)
    message(SEND_ERROR "[${MLIR_LEARN_LOG_PREFIX}|ERROR] >>> ${msg}")
endfunction(log_error msg)

function(log_fatal msg)
    message(FATAL_ERROR "[${MLIR_LEARN_LOG_PREFIX}|FATAL] >>> ${msg}")
endfunction(log_fatal msg)

# This is used for colorizing the output
# For example, to colorize the output with purple, use the following:
# > log_info("${Esc}[0;35mHello, World!${Esc}[m")
string(ASCII 27 Esc)

set(LOG_PREFIX "MLIR-LEARN")

if(STDOUT_IS_TERMINAL)
    set(LOG_RED "${Esc}[0;31m")
    set(LOG_GREEN "${Esc}[0;32m")
    set(LOG_YELLOW "${Esc}[0;33m")
    set(LOG_BLUE "${Esc}[0;34m")
    set(LOG_PURPLE "${Esc}[0;35m")
    set(LOG_CYAN "${Esc}[0;36m")
    set(LOG_WHITE "${Esc}[0;37m")
    set(LOG_RESET "${Esc}[m")
else()
    set(LOG_RED "")
    set(LOG_GREEN "")
    set(LOG_YELLOW "")
    set(LOG_BLUE "")
    set(LOG_PURPLE "")
    set(LOG_CYAN "")
    set(LOG_WHITE "")
    set(LOG_RESET "")
endif()

function(log_info msg)
    message(STATUS "[${LOG_PREFIX}|${LOG_GREEN}INFO${LOG_RESET}] >>> ${msg}")
endfunction(log_info msg)

function(log_warning msg)
    message(WARNING "[${LOG_PREFIX}|WARNING] >>> ${msg}")
endfunction(log_warning msg)

function(log_error msg)
    message(SEND_ERROR "[${LOG_PREFIX}|ERROR] >>> ${msg}")
endfunction(log_error msg)

function(log_fatal msg)
    message(FATAL_ERROR "[${LOG_PREFIX}|FATAL] >>> ${msg}")
endfunction(log_fatal msg)

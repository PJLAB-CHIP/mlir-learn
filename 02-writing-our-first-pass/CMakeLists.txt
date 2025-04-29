####################################################################################################
# @file    "02-writing-our-first-pass/CMakeLists.txt"
# --------------------------------------------------------------------------------------------------
# @brief   This CMake file is used to define the 02-writing-our-first-pass project.
####################################################################################################

project(02-writing-our-first-pass)
log_info("${LOG_PURPLE}Project: ${PROJECT_NAME}${LOG_RESET}")

# This prefix will be added before all target names under this project.
set(PROJECT_TARGET_PREFIX "10")

add_subdirectory("lib")
add_subdirectory("tools")
####################################################################################################
# @file    "03-using-table-gen-for-passes/CMakeLists.txt"
# --------------------------------------------------------------------------------------------------
# @brief   This CMake file is used to define the 03-using-table-gen-for-passes project.
####################################################################################################

project(03-using-table-gen-for-passes)
log_info("${LOG_PURPLE}Project: ${PROJECT_NAME}${LOG_RESET}")

# This prefix will be added before all target names under this project.
set(PROJECT_TARGET_PREFIX "03")

add_subdirectory("include")
add_subdirectory("lib")
add_subdirectory("tools")
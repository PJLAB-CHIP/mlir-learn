####################################################################################################
# @file    "05-using-traits/CMakeLists.txt"
# --------------------------------------------------------------------------------------------------
# @brief   This CMake file is used to define the 05-using-traits project.
####################################################################################################

project(05-using-traits)
log_info("${LOG_PURPLE}Project: ${PROJECT_NAME}${LOG_RESET}")

# This prefix will be added before all target names under this project.
set(PROJECT_TARGET_PREFIX "05")

add_subdirectory("include")
add_subdirectory("lib")
add_subdirectory("tools")
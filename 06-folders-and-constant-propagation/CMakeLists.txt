####################################################################################################
# @file    "06-folders-and-constant-propagation/CMakeLists.txt"
# --------------------------------------------------------------------------------------------------
# @brief   This CMake file is used to define the 06-folders-and-constant-propagation project.
####################################################################################################

project("06-folders-and-constant-propagation")
log_info("${LOG_PURPLE}Project: ${PROJECT_NAME}${LOG_RESET}")

# This prefix will be added before all target names under this project.
set(PROJECT_TARGET_PREFIX "06")

add_subdirectory("include")
add_subdirectory("lib")
add_subdirectory("tools")
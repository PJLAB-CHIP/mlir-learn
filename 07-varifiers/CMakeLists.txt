####################################################################################################
# @file    "07-varifiers/CMakeLists.txt"
# --------------------------------------------------------------------------------------------------
# @brief   This CMake file is used to define the 07-varifiers project.
####################################################################################################

project("07-varifiers")
log_info("${LOG_PURPLE}Project: ${PROJECT_NAME}${LOG_RESET}")

# This prefix will be added before all target names under this project.
set(PROJECT_TARGET_PREFIX "07")

add_subdirectory("include")
add_subdirectory("lib")
add_subdirectory("tools")
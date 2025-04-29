####################################################################################################
# @file    "09-dialect-conversion/CMakeLists.txt"
# --------------------------------------------------------------------------------------------------
# @brief   This CMake file is used to define the 09-dialect-conversion project.
####################################################################################################

project("09-dialect-conversion")
log_info("${LOG_PURPLE}Project: ${PROJECT_NAME}${LOG_RESET}")

# This prefix will be added before all target names under this project.
set(PROJECT_TARGET_PREFIX "09")
# Set the output directory for the project.
set(CMAKE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/${PROJECT_NAME})
log_info("|- Output Directory: ${CMAKE_BINARY_DIR}/${PROJECT_NAME}")

add_subdirectory("include")
add_subdirectory("lib")
add_subdirectory("tools")
####################################################################################################
# @file    "05-using-traits/CMakeLists.txt"
# --------------------------------------------------------------------------------------------------
# @brief   This CMake file is used to define the 05-using-traits project.
####################################################################################################

project(05-using-traits)
mlir_learn_log_info("${MLIR_LEARN_LOG_PURPLE}Project: ${PROJECT_NAME}${MLIR_LEARN_LOG_RESET}")

# This project-wise prefix is intended to be added to all target names.
set(PROJECT_TARGET_PREFIX "05")
# Set the output directory for the project.
set(CMAKE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/${PROJECT_NAME})
mlir_learn_log_info("|- Output Directory: ${CMAKE_OUTPUT_DIR}")

add_subdirectory("include")
add_subdirectory("lib")
add_subdirectory("tools")
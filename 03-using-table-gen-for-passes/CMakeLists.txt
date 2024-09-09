####################################################################################################
# @file    "03-using-table-gen-for-passes/CMakeLists.txt"
# --------------------------------------------------------------------------------------------------
# @brief   This CMake file is used to define the 03-using-table-gen-for-passes project.
####################################################################################################

project(03-using-table-gen-for-passes)
mlir_learn_log_info("${MLIR_LEARN_LOG_PURPLE}Project: ${PROJECT_NAME}${MLIR_LEARN_LOG_RESET}")

# This project-wise prefix is intended to be added to all target names.
set(PROJECT_TARGET_PREFIX "03")
# Set the output directory for the project.
set(CMAKE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/${PROJECT_NAME})


add_subdirectory("include")
add_subdirectory("lib")
add_subdirectory("tools")
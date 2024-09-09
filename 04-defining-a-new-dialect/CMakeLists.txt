####################################################################################################
# @file    "04-defining-a-new-dialect/CMakeLists.txt"
# --------------------------------------------------------------------------------------------------
# @brief   This CMake file is used to define the 04-defining-a-new-dialect project.
####################################################################################################

project(04-defining-a-new-dialect)
mlir_learn_log_info("${MLIR_LEARN_LOG_PURPLE}Project: ${PROJECT_NAME}${MLIR_LEARN_LOG_RESET}")

# This project-wise prefix is intended to be added to all target names.
set(PROJECT_TARGET_PREFIX "04")
# Set the output directory for the project.
set(CMAKE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/${PROJECT_NAME})
mlir_learn_log_info("|- Output Directory: ${CMAKE_OUTPUT_DIR}")

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_OUTPUT_DIR}/include
)

add_subdirectory("include")
add_subdirectory("lib")
add_subdirectory("tools")
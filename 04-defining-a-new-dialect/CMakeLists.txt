####################################################################################################
# @file    "04-defining-a-new-dialect/CMakeLists.txt"
# --------------------------------------------------------------------------------------------------
# @brief   This CMake file is used to define the 04-defining-a-new-dialect project.
####################################################################################################

project(04-defining-a-new-dialect)
log_info("${LOG_PURPLE}Project: ${PROJECT_NAME}${LOG_RESET}")

# This prefix will be added before all target names under this project.
set(PROJECT_TARGET_PREFIX "04")

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/include
)

add_subdirectory("include")
add_subdirectory("lib")
add_subdirectory("tools")
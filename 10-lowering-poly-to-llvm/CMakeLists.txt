####################################################################################################
# @file    "10-lowering-poly-to-llvm/CMakeLists.txt"
# --------------------------------------------------------------------------------------------------
# @brief   This CMake file is used to define the 10-lowering-poly-to-llvm project.
####################################################################################################

project("10-lowering-poly-to-llvm")
log_info("${LOG_PURPLE}Project: ${PROJECT_NAME}${LOG_RESET}")

# This prefix will be added before all target names under this project.
set(PROJECT_TARGET_PREFIX "10")

add_subdirectory("include")
add_subdirectory("lib")
add_subdirectory("tools")
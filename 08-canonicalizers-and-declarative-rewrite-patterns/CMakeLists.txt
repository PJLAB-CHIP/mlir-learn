####################################################################################################
# @file    "08-canonicalizers-and-declarative-rewrite-patterns/CMakeLists.txt"
# --------------------------------------------------------------------------------------------------
# @brief   This CMake file is used to define the 08-canonicalizers-and-declarative-rewrite-patterns 
#          project.
####################################################################################################

project("08-canonicalizers-and-declarative-rewrite-patterns")
log_info("${LOG_PURPLE}Project: ${PROJECT_NAME}${LOG_RESET}")

# This prefix will be added before all target names under this project.
set(PROJECT_TARGET_PREFIX "08")

add_subdirectory("include")
add_subdirectory("lib")
add_subdirectory("tools")
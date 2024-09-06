project(02-writing-our-first-pass)
log_info("${LOG_PURPLE}Project: ${PROJECT_NAME}${LOG_RESET}")

set(CMAKE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/${PROJECT_NAME})

add_subdirectory("lib")
add_subdirectory("tools")
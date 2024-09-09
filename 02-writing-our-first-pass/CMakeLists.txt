project(02-writing-our-first-pass)
mlir_learn_log_info("${MLIR_LEARN_LOG_PURPLE}Project: ${PROJECT_NAME}${MLIR_LEARN_LOG_RESET}")

set(CMAKE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/${PROJECT_NAME})

add_subdirectory("lib")
add_subdirectory("tools")
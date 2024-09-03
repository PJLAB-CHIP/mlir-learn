project(02-writing-our-first-pass)

set(CMAKE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/${PROJECT_NAME})
include_directories(${PROJECT_SOURCE_DIR}/include)


add_subdirectory("lib")
add_subdirectory("tools")
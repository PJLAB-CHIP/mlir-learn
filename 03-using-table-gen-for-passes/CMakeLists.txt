project(03-using-table-gen-for-passes)

set(CMAKE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/${PROJECT_NAME})

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_OUTPUT_DIR}/include)

add_subdirectory("include")
add_subdirectory("lib")
add_subdirectory("tools")
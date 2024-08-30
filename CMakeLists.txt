cmake_minimum_required(VERSION 3.30)

project("MLIR-LEARN")

set(GLOBAL_NAMESPACE "mlir-learn")

include(${CMAKE_SOURCE_DIR}/cmake/Logging.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Compiler.cmake)

add_subdirectory("02-writing-our-first-pass")

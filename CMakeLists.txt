cmake_minimum_required(VERSION 3.30)

project("MLIR-LEARN")

include(${CMAKE_SOURCE_DIR}/cmake/Logging.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Compiler.cmake)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

log_info("Using MLIRConfig.cmake in: ${MLIR_DIR}")
log_info("Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory("standalone")
add_subdirectory("02-writing-our-first-pass")
add_subdirectory("03-using-table-gen-for-passes")
add_subdirectory("04-defining-a-new-dialect")


cmake_minimum_required(VERSION 3.30)

project("MLIR-LEARN")

include(${CMAKE_SOURCE_DIR}/cmake/logging.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/compiler-configs.cmake)

# The following are standard steps for building a MLIR Project.
# 1. Find MLIR, and automatically find LLVM:
find_package(MLIR REQUIRED CONFIG)
log_info("Using MLIRConfig.cmake in: ${MLIR_DIR}")
log_info("Using LLVMConfig.cmake in: ${LLVM_DIR}")
# 2. Append the LLVM and MLIR CMake module path
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
# 3. Include some modules provided by MLIR, where some functions and macros are defined:
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
# 4. For all targest in the subdirectories, include LLVM and MLIR headers, link directories, and add definitions:
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

if(NOT DEFINED PROJECT_NO OR PROJECT_NO STREQUAL "")
    log_fatal("PROJECT_NO is not defined. Please define it in the command line.")
endif()
math(EXPR PROJECT_NO "${PROJECT_NO} - 1")
set(PROJECT_LIST
    "01-standalone"
    "02-writing-our-first-pass"
    "03-using-table-gen-for-passes"
    "04-defining-a-new-dialect"
    "05-using-traits"
    "06-folders-and-constant-propagation"
    "07-varifiers"
    "08-canonicalizers-and-declarative-rewrite-patterns"
    "09-dialect-conversion"
    "10-lowering-poly-to-llvm"
)
list(GET PROJECT_LIST ${PROJECT_NO} TARGET_PROJECT)
# Build only one target specified by PROJECT_LIST. This can reduce clangd indexing time.
add_subdirectory(${TARGET_PROJECT})

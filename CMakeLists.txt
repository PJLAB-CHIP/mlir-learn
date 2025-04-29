cmake_minimum_required(VERSION 3.30)

project("MLIR-LEARN")

include(${CMAKE_SOURCE_DIR}/cmake/Logging.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Compiler.cmake)

find_package(MLIR REQUIRED CONFIG)

mlir_learn_log_info("Using MLIRConfig.cmake in: ${MLIR_DIR}")
mlir_learn_log_info("Using LLVMConfig.cmake in: ${LLVM_DIR}")

# set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
# set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
# set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

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

if(NOT DEFINED TARGET_NO OR TARGET_NO STREQUAL "")
    log_fatal("TARGET_NO is not defined. Please define it in the command line.")
endif()

math(EXPR TARGET_NO "${TARGET_NO} - 1")

set(TARGET_LIST
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

list(GET TARGET_LIST ${TARGET_NO} TARGET)

add_subdirectory(${TARGET})

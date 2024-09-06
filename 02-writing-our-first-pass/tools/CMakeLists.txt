####################################################################################################
# @file    "02-writing-our-fist-pass/tools/CMakeLists.txt"
# --------------------------------------------------------------------------------------------------
# @brief   This CMake file is used to define the ${PROJECT_TARGET_PREFIX}-tutorial-opt executable.
####################################################################################################

set(TARGET_NAME "02-tutorial-opt")
log_info("|- Target <exe>: ${TARGET_NAME}")

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(
    LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRArithDialect
    MLIROptLib
    02-tutorial-lib
)

add_llvm_executable(${TARGET_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/tutorial-opt.cpp)

set_target_properties(${TARGET_NAME} PROPERTIES 
    # Output Directory:
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/tools
)

target_link_libraries(${TARGET_NAME} PRIVATE ${LIBS})

llvm_update_compile_flags(${TARGET_NAME})
target_link_libraries(${TARGET_NAME} PRIVATE ${LIBS})

mlir_check_all_link_libraries(${TARGET_NAME})
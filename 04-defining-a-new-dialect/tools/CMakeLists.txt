set(TARGET_NAME "04-tutorial-opt")

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRArithDialect
    MLIROptLib
    ${GLOBAL_NAMESPACE}::04-tutorial-lib
    )

add_llvm_executable(${TARGET_NAME} tutorial-opt.cpp)

set_target_properties(${TARGET_NAME} PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/tools
)

llvm_update_compile_flags(${TARGET_NAME})
target_link_libraries(${TARGET_NAME} PRIVATE ${LIBS})

mlir_check_all_link_libraries(${TARGET_NAME})

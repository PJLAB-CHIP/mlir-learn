set(TARGET_NAME "02-tutorial-opt")


add_executable(${TARGET_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/tutorial-opt.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE ${GLOBAL_NAMESPACE}::02-AffineFullUnroll)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
# get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
# get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(${TARGET_NAME} PRIVATE ${dialect_libs} MLIROptLib)

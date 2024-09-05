set(TARGET_NAME "02-tutorial-opt")


add_executable(${TARGET_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/tutorial-opt.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE 02-AffineFullUnroll)

set_target_properties(${TARGET_NAME} PROPERTIES 
    # Output Directory:
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/tools
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

target_link_libraries(${TARGET_NAME} PRIVATE ${dialect_libs} MLIROptLib)

set(TARGET_NAME "02-AffineFullUnroll")
log_info("Target: ${TARGET_NAME} Library")

add_mlir_dialect_library(
    # Library Name:
    ${TARGET_NAME}

    # Source Files:
    ${CMAKE_CURRENT_SOURCE_DIR}/Transform/Affine/AffineFullUnroll.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Transform/Arith/MulToAdd.cpp

    # Link Libraries:
    LINK_LIBS PUBLIC
    MLIRIR 
    MLIRPass 
    MLIRTransforms 
    MLIRAffineDialect
    MLIRFuncDialect 
    MLIRInferTypeOpInterface
)

set_target_properties(${TARGET_NAME} PROPERTIES 
    # Output Directory:
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/lib
)